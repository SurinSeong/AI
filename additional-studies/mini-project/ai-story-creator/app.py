import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from sqlmodel import Field, Session, SQLModel, create_engine, select
from typing import Optional, List, Tuple
import hashlib
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from PIL import Image as PILImage
import os

# 캐릭터 일관성을 위한 고정 속성 추가 (한국어)
CHARACTER_DESCRIPTION = "young korean man with blue hoodie"

# 데이터베이스 모델
class Story(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)

class ImageCache(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt_hash: str = Field(index=True)
    image_path: str
    created_at: datetime = Field(default_factory=datetime.now)

# 데이터베이스 초기화
engine = create_engine("sqlite:///storybook.db")
SQLModel.metadata.create_all(engine)

# 모델 초기화
print("모델 로딩 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# LLM 모델
llm_model_name = "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

# Stable Diffusion 모델
sd_model_name = "Lykon/DreamShaper"
sd_pipe = StableDiffusionPipeline.from_pretrained(
    sd_model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=False
)
sd_pipe = sd_pipe.to(device)

# 모델 로드 후 스케줄러 변경
sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    sd_pipe.scheduler.config,
    use_karras_sigmas=True,  # Karras schedule
    algorithm_type="dpmsolver++"
)

# 이미지 저장 디렉토리
os.makedirs("generated_images", exist_ok=True)


def generate_story(prompt: str) -> Tuple[str, List[str]]:
    """프롬프트로부터 스토리 생성"""
    system_prompt = f"""당신은 뛰어난 스토리텔러입니다.
다음 주제를 바탕으로, 5개의 문단으로 구성된 흥미로운 이야기를 작성하세요.

규칙:
- 주인공은 '청년' 또는 '그'로만 지칭하세요 (이름 사용 금지)
- 주인공은 안경을 쓴 20대 청년입니다
- 각 문단은 2~4개의 문장으로 구성
- 시각적으로 표현 가능한 구체적인 장면 묘사 포함
- 순수 한국어만 사용
- 각 문단마다 명확한 장소와 행동 묘사

주제: {prompt}

이야기:"""
    
    inputs = tokenizer(system_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.1,
        )
    
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    story = story.replace(system_prompt, "").strip()
    
    # 문단 분리
    paragraphs = []
    raw_paragraphs = story.split("\n\n")
    for p in raw_paragraphs:
        p = p.strip()
        if p and len(p) > 20:
            paragraphs.append(p)
    
    paragraphs = paragraphs[:5]
    
    # DB 저장
    with Session(engine) as session:
        db_story = Story(prompt=prompt, content="\n\n".join(paragraphs))
        session.add(db_story)
        session.commit()
    
    return "\n\n".join(paragraphs), paragraphs


def analyze_text_for_english_scene(text: str, paragraph_num: int = 1) -> str:
    """텍스트를 분석하여 영어 씬 추출 (기본 10개 키워드)"""
    
    # 디버깅용 출력
    print(f"[{paragraph_num}] 텍스트 분석 중: {text[:60]}...")
    
    # 핵심 키워드 10개만 처리
    # 1. 카페 + 노트북/컴퓨터
    if "카페" in text and ("노트북" in text or "컴퓨터" in text):
        return "working on laptop in coffee shop"
    
    # 2. 카페 (일반)
    elif "카페" in text:
        return "in a coffee shop"
    
    # 3. 프로그래밍/코딩
    elif "프로그래밍" in text or "코딩" in text or "코드" in text:
        return "coding on laptop"
    
    # 4. 회의/미팅
    elif "회의" in text or "미팅" in text:
        return "in a meeting"
    
    # 5. 발표/프레젠테이션
    elif "발표" in text or "프레젠테이션" in text:
        return "giving presentation"
    
    # 6. 동료/팀
    elif "동료" in text or "팀" in text:
        return "with team members"
    
    # 7. 성공/축하
    elif "성공" in text or "축하" in text:
        return "celebrating success"
    
    # 8. 계획
    elif "계획" in text:
        return "planning"
    
    # 9. 사무실
    elif "사무실" in text:
        return "in office"
    
    # 10. 투자/투자자
    elif "투자" in text:
        return "meeting investors"
    
    # 기본값 (문단별)
    defaults = {
        1: "young entrepreneur working",
        2: "developing project",
        3: "collaborating with others",
        4: "business presentation",
        5: "successful achievement"
    }
    
    return defaults.get(paragraph_num, "at work")

    
def generate_image(text: str, paragraph_num: int = 1) -> str:
    """텍스트로부터 이미지 생성"""
    # 프롬프트 해시 생성
    prompt_hash = hashlib.md5(text.encode()).hexdigest()
    
    # 캐시 확인
    with Session(engine) as session:
        cached = session.exec(
            select(ImageCache).where(ImageCache.prompt_hash == prompt_hash)
        ).first()
        
        if cached:
            return cached.image_path
    
    # 씬 추출
    print(f"\n[{paragraph_num}/5] 이미지 생성 중...")
    scene = analyze_text_for_english_scene(text)
    
    # 최종 프롬프트 생성 
    final_prompt = f"{CHARACTER_DESCRIPTION} {scene}"
    
    
    print("최종 프롬프트: ", final_prompt)
    print(f"프롬프트 길이: {len(final_prompt)} 글자")
    
    # 네거티브 프롬프트
    negative_prompt = "realistic, photo, multiple people, crowd"
    
    # Seed 고정
    base_seed = 396135060
    # Seed 미세 변화
    seed = base_seed + (paragraph_num * 10)  # 10, 20, 30, 40, 50
    # Seed 다변화
    #text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    #seed = base_seed + (text_hash % 1000)
    generator = torch.Generator(device=device).manual_seed(seed)
    #generator = torch.Generator(device=device).manual_seed(
    #    torch.randint(0, 100000, (1,)).item()
    #)
    
    # 이미지 생성
    with torch.no_grad():
        image = sd_pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=6.0,
            height=512,
            width=512,
            generator=generator,
            safety_checker=None,
            requires_safety_checker=False
        ).images[0]
    
    # 이미지 저장
    image_path = f"generated_images/{prompt_hash}.png"
    image.save(image_path)
    
    # 캐시 저장
    with Session(engine) as session:
        cache_entry = ImageCache(prompt_hash=prompt_hash, image_path=image_path)
        session.add(cache_entry)
        session.commit()
    
    return image_path

def create_pdf(story_text: str, image_paths: List[str], output_path: str = "storybook.pdf"):
    """스토리와 이미지로 PDF 생성"""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []

    font_path = "malgun.ttf"
    pdfmetrics.registerFont(TTFont('맑은고딕', font_path))

    # 스타일 설정
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName="맑은고딕",
        fontSize=24,
        textColor='black',
        alignment=TA_CENTER,
        spaceAfter=30
    )

    text_style = ParagraphStyle(
        'CustomText',
        parent=styles['Normal'],
        fontName="맑은고딕",
        fontSize=12,
        leading=18,
        alignment=TA_JUSTIFY,
        spaceAfter=20
    )

    story.append(Paragraph("AI 스토리북", title_style))
    story.append(Spacer(1, 1*cm))

    paragraphs = story_text.strip().split("\n\n")
    for i, para in enumerate(paragraphs):
        story.append(Paragraph(para.strip(), text_style))
        
        if i < len(image_paths) and os.path.exists(image_paths[i]):
            img = Image(image_paths[i], width=15*cm, height=10*cm)
            story.append(img)
            story.append(Spacer(1, 1*cm))

        if i < len(paragraphs) - 1:
            story.append(PageBreak())

    doc.build(story)
    return output_path

# Gradio 인터페이스
def process_story(prompt: str):
    """스토리 생성 처리"""
    story, paragraphs = generate_story(prompt)
    return story, gr.update(visible=True), paragraphs

def generate_images_batch(paragraphs: List[str]):
    """배치로 이미지 생성 (진행률 표시)"""
    from tqdm import tqdm
    
    image_paths = []
    for i, para in tqdm(enumerate(paragraphs), total=len(paragraphs), desc="이미지 생성"):
        img_path = generate_image(para, paragraph_num=i+1)
        image_paths.append(img_path)
        
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return image_paths

def create_storybook(story_text: str, paragraphs: List[str]):
    """스토리북 PDF 생성"""
    # 이미지 생성
    image_paths = generate_images_batch(paragraphs)
    
    # PDF 생성
    pdf_path = create_pdf(story_text, image_paths)
    
    # 이미지 갤러리용 데이터
    images = [PILImage.open(path) for path in image_paths]
    
    return images, pdf_path

# Gradio UI
with gr.Blocks(title="AI 스토리북 저작 도구") as app:
    gr.Markdown("# AI 스토리북 저작 도구")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="스토리 주제 입력",
                placeholder="예: 스타트업 창업 성공 스토리",
                lines=2
            )
            generate_btn = gr.Button("스토리 생성", variant="primary")
            
            story_output = gr.Textbox(
                label="생성된 스토리",
                lines=15,
                interactive=True
            )
            
            create_book_btn = gr.Button(
                "스토리북 생성 (이미지 + PDF)",
                variant="secondary",
                visible=False
            )
        
        with gr.Column():
            image_gallery = gr.Gallery(
                label="생성된 이미지",
                show_label=True,
                elem_id="gallery",
                columns=2,
                rows=3,
                height="auto"
            )
            
            pdf_output = gr.File(
                label="PDF 다운로드",
                visible=True
            )
    
    # 상태 저장
    paragraphs_state = gr.State([])
    
    # 이벤트 핸들러
    generate_btn.click(
        fn=process_story,
        inputs=[prompt_input],
        outputs=[story_output, create_book_btn, paragraphs_state]
    )
    
    create_book_btn.click(
        fn=create_storybook,
        inputs=[story_output, paragraphs_state],
        outputs=[image_gallery, pdf_output]
    )

if __name__ == "__main__":
    app.launch(share=True)