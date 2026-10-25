# 배경 제거 (`promboard.remove-bg`)

PromBoard 공식 코어 플러그인입니다. 캔버스에 **배경 제거** 노드를 더해, 연결한 이미지에서 주인공만 남기고 배경을 지운 투명 PNG 를 만듭니다. 노드에서 원본과 결과를 전/후 비교 슬라이더로 봅니다.

## 설치

PromBoard 앱이 처음 켜질 때 공식 팩과 함께 자동으로 받습니다. 직접 받으려면 앱의 **플러그인 → 코어 플러그인**에서 받습니다. 코드가 든 플러그인이라 받은 뒤 켜야 돌고, AI 모델은 실행 준비 때 크기를 보여 주고 확인을 받은 뒤 받습니다.

## 설정

| 칸 | 값 | 뜻 |
|---|---|---|
| 품질 | 빠름(처음 값) · 정밀 | 빠름은 ISNet — RAM 약 1.6GB, 배경이 복잡하면 경계가 번질 수 있다. 정밀은 BiRefNet_lite — 복잡한 배경에서도 깔끔하지만 RAM 6~7GB |
| 가장자리 | 부드럽게(처음 값) · 또렷하게 | 부드럽게는 머리카락 같은 반투명 경계를 살리고, 또렷하게는 경계를 딱 잘라 남긴다 |

그래픽카드 없이 CPU 로 돕니다. 맥은 Apple Silicon · macOS 14 이상만 됩니다.

## AI 모델

모델 파일은 이 저장소와 zip 에 들어 있지 않습니다. 고른 품질의 모델만 원래 출처에서 받고 SHA-256 으로 확인합니다(`manifest.json` 의 `requires.models`).

| 품질 | 모델 | 출처 | 라이선스 | 크기 |
|---|---|---|---|---|
| 빠름 | ISNet (`isnet-general-use.onnx`) | [danielgatis/rembg](https://github.com/danielgatis/rembg/releases/tag/v0.0.0) | Apache-2.0 | 약 179MB |
| 정밀 | BiRefNet_lite (`BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx`) | [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet/releases/tag/v1) | MIT | 약 224MB |

## 릴리스

Release 의 zip 은 PromBoard 앱 저장소의 `server/src/tools/packPlugin.ts` 로 만들고, 레지스트리([promboard-app/registry](https://github.com/promboard-app/registry))에 등록합니다. 한 번 올린 버전은 고치지 않고, 바꿀 때는 버전을 올립니다.

## 라이선스

코드는 [MIT](LICENSE). 모델 라이선스는 위 표를 따릅니다.
