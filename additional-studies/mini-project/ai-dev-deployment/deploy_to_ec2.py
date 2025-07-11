import paramiko
from scp import SCPClient
import yaml
from pathlib import Path

# 설정 파일 로드
with open("deploy-config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ec2_ip = config["ec2"]["ip"]
ec2_user = config["ec2"]["user"]
key_path = config["ec2"]["key_path"]

local_project_dir = Path(config["paths"]["local_project_dir"]).resolve()
remote_dir = config["paths"]["remote_dir"]

def create_ssh_client(ip, user, key_file):
    key = paramiko.RSAKey.from_private_key_file(key_file)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=user, pkey=key)
    return ssh

def deploy():
    ssh = create_ssh_client(ec2_ip, ec2_user, key_path)

    # 원격 디렉토리 생성
    stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_dir}")
    stdout.channel.recv_exit_status()

    # 소스코드 전송
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(local_project_dir, recursive=True, remote_path=remote_dir)

    print("✅ 배포 완료")
    ssh.close()

if __name__ == "__main__":
    deploy()
