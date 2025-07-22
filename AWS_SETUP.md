# AWS Bedrock 설정 가이드

이 문서는 AWS Bedrock을 사용하기 위한 설정 방법을 안내합니다.

## 목차

- [사전 요구사항](#사전-요구사항)
- [AWS Bedrock 활성화](#aws-bedrock-활성화)
- [모델 액세스 설정](#모델-액세스-설정)
- [IAM 권한 설정](#iam-권한-설정)
- [AWS CLI 구성](#aws-cli-구성)
- [환경 변수 설정](#환경-변수-설정)

## 사전 요구사항

- AWS 계정
- AWS CLI 설치
- Python 3.8 이상

## AWS Bedrock 활성화

1. [AWS Management Console](https://console.aws.amazon.com/)에 로그인합니다.
2. 검색창에 "Bedrock"을 입력하고 서비스를 선택합니다.
3. AWS Bedrock 서비스에 처음 접근하는 경우, 서비스 활성화 과정을 진행합니다.
4. 리전이 Bedrock을 지원하는지 확인하세요. (예: us-east-1, us-west-2)

## 모델 액세스 설정

AWS Bedrock에서 사용할 모델에 대한 액세스 권한을 설정해야 합니다:

1. AWS Bedrock 콘솔에서 왼쪽 메뉴의 "Model access"를 선택합니다.
2. "Manage model access"를 클릭합니다.
3. 사용하려는 모델(예: Claude, Titan)을 선택하고 "Request model access"를 클릭합니다.
4. 액세스 요청이 승인되면 해당 모델을 사용할 수 있습니다.

## IAM 권한 설정

AWS Bedrock을 사용하기 위해 필요한 IAM 권한을 설정합니다:

1. [IAM 콘솔](https://console.aws.amazon.com/iam/)로 이동합니다.
2. 사용자 또는 역할을 선택합니다.
3. "권한 추가" > "정책 연결"을 클릭합니다.
4. 다음 권한이 포함된 정책을 연결합니다:
   - `bedrock:InvokeModel`
   - `bedrock:InvokeModelWithResponseStream`
   - `bedrock:GetFoundationModel`
   - `bedrock:ListFoundationModels`

또는 다음과 같은 인라인 정책을 생성할 수 있습니다:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:GetFoundationModel",
                "bedrock:ListFoundationModels"
            ],
            "Resource": "*"
        }
    ]
}
```

## AWS CLI 구성

AWS CLI를 구성하여 AWS Bedrock에 접근할 수 있도록 합니다:

```bash
aws configure
```

프롬프트에 따라 다음 정보를 입력합니다:
- AWS Access Key ID
- AWS Secret Access Key
- Default region name (예: us-east-1)
- Default output format (json)

## 환경 변수 설정

애플리케이션에서 AWS 자격 증명을 사용하기 위해 환경 변수를 설정할 수 있습니다:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=your_region  # 예: us-east-1
```

또는 `.env` 파일을 생성하여 환경 변수를 설정할 수 있습니다:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
```

그리고 Python 코드에서 다음과 같이 로드합니다:

```python
from dotenv import load_dotenv
load_dotenv()
```

## 확인 방법

설정이 올바르게 되었는지 확인하기 위해 다음 AWS CLI 명령을 실행해볼 수 있습니다:

```bash
aws bedrock list-foundation-models --region us-east-1
```

이 명령이 성공적으로 실행되면 사용 가능한 Bedrock 모델 목록이 표시됩니다.
