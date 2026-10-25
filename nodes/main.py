"""test.remove-bg — 시험 플러그인 A: 배경 제거 (Scrum 4 S2-2)

version: 0.3.0 (manifest 와 같다) | lastupdated: 2026-10-14
v0.1.1 (Scrum 4 T10): 세션 캐시 — 맥에서 세션을 만드는 동안(CoreML 변환) 하트비트가 45초 끊겨 hang 으로 죽던 것의 후속.
  만든 결과를 ctx.cache_dir 에 남겨 두 번째 사이드카부터 빨리 뜬다. 아래 「세션 캐시」.
v0.1.2 (T10 맥 실측 2026-10-13, PM 결정): 맥(mps)도 CPU 만 쓴다. CoreML(MLProgram) 변환이 이 모델의 deformable conv 에서
  실패하고(`Unable to parse ML Program: … aspp_deforms … Conv: Required param 'pad' is missing`) 매번 26~134초를 쓴 뒤
  CPU 로 돌아갔다. 맥 CPU 는 모델 읽기 0.9~1.8초 · 한 장 11~17초, 결과 알파는 같다.
v0.2.0 (00 PM 결정 25): 설정 '품질' — 정밀 = BiRefNet_lite, 빠름 = ISNet. '자동'(기본값)은 서버가 이 PC 메모리로 풀어서
  넘긴다(16GB 이상이면 정밀, 아니면 빠름 — manifest 의 options[].auto). 모델은 고른 쪽만 받는다(manifest 모델의 when).
v0.3.0 (PM 2026-10-14, PR #20 질문 1번 답): '품질'은 정밀 · 빠름 둘, 처음 값은 빠름. '자동'은 뺐다 — 결과가 마음에 안
  들면 사용자가 정밀로 다시 돌린다. 노드 동작은 그대로다(manifest 만 바뀜).

모델(MODELS — 품질마다 이름 · 전처리 · 출력 읽는 법)
  - 정밀(precise): BiRefNet_lite 공식 ONNX 파일(ZhengPeng7/BiRefNet GitHub 릴리스 v1, MIT). 사진 한 장에 RAM 6~7GB.
  - 빠름(fast): ISNet general-use ONNX(DIS, Apache-2.0 — rembg GitHub 릴리스의 변환본). RAM 약 1.6GB · 약 2초.
    배경이 복잡하면 경계가 번질 수 있다(docs/web-research/로컬 전환 기능 리서치/10_배경-제거-모델-플러그인-후보.md).
  manifest 의 requires.models(url 꼴)로 실행 준비 때 원래 출처에서 받고, 노드는 ctx.model_path(이름) 로 읽는다.
  모르는 품질 값(예: 0.2.0 의 'auto')이면 가벼운 빠름으로 돈다 — 8GB PC 에서 멈추지 않게.
실행: onnxruntime. torch 를 쓰지 않는다("torch": "none").
  - 실행 장치(provider)는 ctx.device 를 따른다. cpu 면 CPU 만 쓴다.
    cuda 면 CUDA·DirectML 을 앞에 둔다. 단 설치된 onnxruntime 에 그 장치가 있을 때만이다.
    잠금 파일은 CPU 판 onnxruntime 이라 CUDA·DirectML 은 보통 없다.
    mps(Apple Silicon mac)도 CPU 만 쓴다 — mac 판에는 CoreML 이 들어 있지만 이 모델은 변환이 실패한다(v0.1.2).
  - 가속 장치(CUDA 등)로 세션을 만들거나 돌리다 실패하면 CPU 로 한 번 다시 한다.
  - 메모리: 세션을 만들 때 CPU 메모리 아레나를 끈다(1024px 한 장 최대 RAM 약 12GB → 6~7GB, T6 실측).
  - 세션은 프로세스마다 하나 만들어 둔다. 사이드카가 떠 있는 동안 같은 품질의 두 번째 장부터는 모델 읽기 없이 추론만
    한다. 품질을 바꾸면 그 모델로 세션을 새로 만들고 앞 세션은 버린다(두 모델을 같이 메모리에 올리지 않는다).
  - 세션을 만들기 바로 앞에 ctx.progress(…, "load-model") 를 보낸다. onnxruntime 이 세션을 만드는 내내 GIL 을 쥐어
    하트비트가 끊기는데, 서버는 이 단계 동안 하트비트 없이 더 오래 기다린다(server/src/localai/sidecar.ts).
세션 캐시(ctx.cache_dir 이 있을 때만 — None 이면 지금처럼 캐시 없이)
  - CPU: 원본을 열며 최적화한 그래프를 `<모델 줄기>-<원본 sha256 앞 16자>-ort<버전>-cpu.opt.onnx` 로 저장해 두고, 다음부터
    그 파일을 최적화 없이(ORT_DISABLE_ALL) 연다. 임시 이름으로 쓴 뒤 os.replace 로 옮겨 덜 쓴 파일이 최종 이름으로 남지
    않는다. 저장본이 열리지 않으면 지우고 원본으로 다시 만든다. 저장은 ORT_ENABLE_EXTENDED 까지만 한다 — ORT_ENABLE_ALL 의
    레이아웃 최적화(NCHWc)는 저장한 PC 의 CPU 명령(AVX2 등)에 묶인다(공식 문서
    https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#onlineoffline-mode).
  - CoreML(지금은 쓰지 않는다 — ACCEL_BY_DEVICE 의 mps 에 다시 넣으면 이 옵션으로 연다): CoreML EP 옵션 ModelCacheDirectory 로 변환한 CoreML 모델을 `coreml/ort<버전>/` 에 남긴다. EP 는 모델이
    바뀐 것을 알아채지 못하고 캐시 키를 모델 경로 해시로 만드는데, 모델 경로가 `_plugin-models/_url/<sha256>/…`
    (내용 주소)라 모델이 바뀌면 키도 바뀐다. 옵션을 모르는 onnxruntime 이면 세션 만들기가 실패해 CPU 로 다시 한다.
처리 순서(두 모델 모두 공식 추론과 같다)
  1. 원본을 EXIF 방향대로 세우고 RGB 로 바꾼 뒤 1024×1024 로 줄인다.
  2. 255 로 나누고 모델의 평균·표준편차로 정규화해 NCHW float32 로 넣는다
     (BiRefNet = ImageNet 평균·표준편차, ISNet = 평균 0.5 · 표준편차 1).
  3. 마스크를 만들고 원본 크기로 키운다. BiRefNet 은 마지막 출력(로짓)에 시그모이드, ISNet 은 첫 출력(0~1)을
     최소·최대로 다시 늘린다(DIS 공식 추론).
  4. 마스크를 알파로 붙여 RGBA PNG 로 저장한다(원본에 알파가 있으면 곱한다). 완전히 투명한 곳의 색은 0 으로 둔다.
설정 quality: precise(정밀) · fast(빠름, 처음 값).
설정 edge: soft = 마스크를 그대로 알파로(머리카락 같은 반투명 경계를 살린다) · sharp = 0.5 를 기준으로 0·255 로 자른다.
진행률: 읽기 → 모델 읽기(처음 한 번) → 추론(지난 추론 시간으로 어림한 진행률) → 합치기 → 저장.
취소: 추론 중에도 0.2초마다 ctx.cancelled 를 보고, 취소면 onnxruntime 실행을 멈춘다(RunOptions.terminate).
노드 코드 약속은 docs/Scrum4/02_스키마-v0.md §6 이다. numpy·Pillow·onnxruntime 은 함수 안에서 불러온다
(실행 준비의 마지막 시험(--check)이 가볍게 끝나고, 시험 코드가 numpy 만으로 전처리·후처리를 볼 수 있게).
"""
import json
import math
import os
import re
import threading
import time

MODEL_NAME = "birefnet/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
SIZE = 1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EDGES = ("soft", "sharp")
# 품질 → 모델. name = manifest requires.models[].path(= ctx.models 키), stem = 세션 캐시 파일 이름 앞부분,
# output = 마스크가 든 출력 자리, mask = "logits"(시그모이드) | "probs"(0~1, 최소·최대로 늘림), run_sec = 진행률 어림 첫 값.
MODELS = {
    "precise": {"name": MODEL_NAME, "stem": "birefnet", "mean": MEAN, "std": STD, "output": -1, "mask": "logits", "run_sec": 20.0},
    "fast": {"name": "isnet/isnet-general-use.onnx", "stem": "isnet", "mean": (0.5, 0.5, 0.5), "std": (1.0, 1.0, 1.0),
             "output": 0, "mask": "probs", "run_sec": 3.0},
}
QUALITIES = ("precise", "fast")
FALLBACK_QUALITY = "fast"
OUT_NAME = "remove-bg.png"

CPU_EP = "CPUExecutionProvider"
COREML_EP = "CoreMLExecutionProvider"
ACCEL_BY_DEVICE = {
    "cuda": ("CUDAExecutionProvider", "DmlExecutionProvider"),
    # 맥도 CPU(PM 2026-10-13). CoreML 은 이 모델 변환이 실패해 매번 시간만 쓰고 CPU 로 돌아갔다(머리 주석 v0.1.2).
    "mps": (),
}
WATCH_SEC = 0.2

CACHE_STEM = "birefnet"
CPU_CACHE_SUFFIX = "-cpu.opt.onnx"
# 저장할 때의 최적화 단계. ORT_ENABLE_ALL 은 레이아웃 최적화까지 해 저장본이 그 PC 의 CPU 명령에 묶인다(위 공식 문서).
CPU_SAVE_LEVEL = "ORT_ENABLE_EXTENDED"
# CoreML EP 옵션(onnxruntime 1.30). 이름·값은 공식 문서
#   https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#available-options-new-api
# 와 v1.30.0 소스(include/onnxruntime/core/providers/coreml/coreml_provider_factory.h ·
# onnxruntime/core/providers/coreml/coreml_options.cc)로 확인했다(2026-10-12).
#   ModelFormat          MLProgram(Core ML 5+, macOS 12+) | NeuralNetwork(기본)
#   MLComputeUnits       ALL(기본) | CPUOnly | CPUAndGPU | CPUAndNeuralEngine
#   ModelCacheDirectory  변환한 CoreML 모델을 남길 폴더(없으면 만든다). 비우면 세션마다 다시 변환한다(기본)
# 모르는 옵션 이름·값이면 세션 만들기가 실패한다 → _get_session 의 'CPU 로 다시' 길.
COREML_OPTIONS = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_CACHE = {}
# 품질마다 지난 추론 시간(진행률 어림).
_STATS = {q: spec["run_sec"] for q, spec in MODELS.items()}


# ---------------------------------------------------------------------------
# 전처리 · 후처리 (numpy 만)
# ---------------------------------------------------------------------------


def preprocess(rgb, mean=MEAN, std=STD):
    """1024×1024 RGB uint8 배열(H, W, 3) → 모델 입력 (1, 3, H, W) float32. 기본은 BiRefNet(ImageNet) 정규화."""
    import numpy as np

    a = np.asarray(rgb, dtype=np.float32) / 255.0
    a = (a - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def sigmoid(x):
    """넘침 없는 시그모이드(float32). 로짓 범위가 −35 ~ +89 쯤이다."""
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    return (0.5 * (np.tanh(0.5 * x) + 1.0)).astype(np.float32)


def logits_to_mask(logits):
    """모델 출력 (1, 1, H, W) 로짓 → (H, W) 0~1 마스크."""
    import numpy as np

    a = np.asarray(logits, dtype=np.float32)
    while a.ndim > 2:
        a = a[0]
    return sigmoid(a)


def probs_to_mask(probs):
    """ISNet 출력 (1, 1, H, W) 0~1 값 → (H, W) 마스크. 최소·최대로 0~1 에 다시 늘린다(DIS 공식 추론). 값이 한 가지면 그대로."""
    import numpy as np

    a = np.asarray(probs, dtype=np.float32)
    while a.ndim > 2:
        a = a[0]
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.clip(a, 0.0, 1.0)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def output_to_mask(outputs, spec):
    """세션 출력 목록 → 0~1 마스크. 어느 출력을 어떻게 읽을지는 모델 spec(MODELS) 이 정한다."""
    out = outputs[spec["output"]]
    return logits_to_mask(out) if spec["mask"] == "logits" else probs_to_mask(out)


def quality_of(settings):
    """설정 → (품질, 바꿨나). 모르는 값(예: 0.2.0 의 'auto')이면 가벼운 빠름."""
    q = settings.get("quality")
    return (q, False) if q in MODELS else (FALLBACK_QUALITY, True)


def alpha_from_mask(mask, edge="soft"):
    """0~1 마스크 → 알파 uint8. sharp 는 0.5 를 기준으로 0·255 로 자른다."""
    import numpy as np

    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if edge == "sharp":
        return np.where(m >= 0.5, 255, 0).astype(np.uint8)
    return np.rint(m * 255.0).astype(np.uint8)


def compose_rgba(rgb, alpha, src_alpha=None):
    """RGB (H, W, 3) + 알파 (H, W) → RGBA (H, W, 4) uint8. 원본 알파가 있으면 곱하고, 투명한 곳의 색은 0."""
    import numpy as np

    a = np.asarray(alpha, dtype=np.uint16)
    if src_alpha is not None:
        a = (a * np.asarray(src_alpha, dtype=np.uint16) + 127) // 255
    a = a.astype(np.uint8)
    out = np.empty(a.shape + (4,), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = a
    out[a == 0, :3] = 0
    return out


def pick_providers(available, device):
    """ctx.device 와 설치된 onnxruntime 이 가진 장치로 provider 순서를 정한다. 끝은 늘 CPU."""
    have = set(available or ())
    return [p for p in ACCEL_BY_DEVICE.get(str(device), ()) if p in have] + [CPU_EP]


# ---------------------------------------------------------------------------
# onnxruntime 세션 · 추론
# ---------------------------------------------------------------------------


def _ort():
    import onnxruntime as ort

    return ort


def _session_options(ort, providers):
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.log_severity_level = 3
    if "DmlExecutionProvider" in providers:
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return so


def _ort_tag(ort):
    """캐시 이름에 넣을 onnxruntime 버전(파일 이름에 쓸 수 있는 글자만)."""
    return "ort" + re.sub(r"[^0-9A-Za-z.]", "_", str(getattr(ort, "__version__", "unknown")))


def _model_sha256(model_path, name=MODEL_NAME):
    """원본 모델 sha256. 받은 자리 `_plugin-models/_url/<sha256>/…` 에서, 아니면 manifest requires.models(path = name) 에서. 모르면 None."""
    parts = os.path.normpath(str(model_path)).split(os.sep)
    for i in range(1, len(parts) - 1):
        if parts[i - 1] == "_plugin-models" and parts[i] == "_url" and _SHA256.match(parts[i + 1]):
            return parts[i + 1]
    manifest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "manifest.json")
    try:
        with open(manifest, encoding="utf-8") as f:
            models = json.load(f)["requires"]["models"]
        for m in models:
            if m.get("path") == name and _SHA256.match(str(m.get("sha256", ""))):
                return m["sha256"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return None


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _cpu_cached_session(ort, model_path, cache_dir, sha, log, stem=CACHE_STEM):
    """CPU 세션 + 저장본 캐시 → (세션, "hit"|"miss"). stem = 모델마다 다른 파일 이름 앞부분(MODELS[].stem)."""
    name = f"{stem}-{sha[:16]}-{_ort_tag(ort)}{CPU_CACHE_SUFFIX}"
    final = os.path.join(cache_dir, name)
    if os.path.isfile(final):
        so = _session_options(ort, [CPU_EP])
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        try:
            return ort.InferenceSession(final, sess_options=so, providers=[CPU_EP]), "hit"
        except Exception as exc:
            log(f"cached model {name} did not open ({type(exc).__name__}: {exc}); rebuilding", "warn")
            _remove_quietly(final)
    try:
        names = os.listdir(cache_dir)
    except OSError:
        names = []  # 폴더가 그사이 사라졌으면 아래 저장이 실패해 캐시 없이 연다
    for old in names:
        # 끊겨 남은 임시 파일 · 같은 줄기의 다른 원본이나 onnxruntime 버전 저장본(사이드카는 플러그인마다 하나라 겹치지 않는다).
        # 줄기가 다른 모델(정밀 ↔ 빠름)의 저장본은 지우지 않는다 — 품질을 바꿔도 각자 캐시가 남는다.
        stale_tmp = old.startswith(f".{stem}-") and old.endswith(".tmp")
        other_copy = old.startswith(f"{stem}-") and old.endswith(CPU_CACHE_SUFFIX) and old != name
        if stale_tmp or other_copy:
            _remove_quietly(os.path.join(cache_dir, old))
    tmp = os.path.join(cache_dir, f".{name}.{os.getpid()}.tmp")
    so = _session_options(ort, [CPU_EP])
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, CPU_SAVE_LEVEL)
    so.optimized_model_filepath = tmp
    try:
        session = ort.InferenceSession(model_path, sess_options=so, providers=[CPU_EP])
    except Exception as exc:
        _remove_quietly(tmp)
        log(f"saving the optimized model failed ({type(exc).__name__}: {exc}); opening without cache", "warn")
        return ort.InferenceSession(model_path, sess_options=_session_options(ort, [CPU_EP]), providers=[CPU_EP]), "off"
    try:
        os.replace(tmp, final)
    except OSError as exc:
        _remove_quietly(tmp)
        log(f"could not keep the optimized model ({exc})", "warn")
    return session, "miss"


def _new_session(model_path, providers, cache_dir=None, log=None, spec=None):
    """세션 하나 → (세션, 캐시 "hit"|"miss"|"off"). cache_dir 이 None 이면 캐시 없이. spec 이 없으면 정밀(BiRefNet)."""
    ort = _ort()
    providers = list(providers)
    log = log or (lambda message, level="info": None)
    spec = spec or MODELS["precise"]
    if cache_dir and providers == [CPU_EP]:
        sha = _model_sha256(model_path, spec["name"])
        if sha:
            return _cpu_cached_session(ort, model_path, cache_dir, sha, log, spec["stem"])
    so = _session_options(ort, providers)
    if COREML_EP not in providers:
        return ort.InferenceSession(model_path, sess_options=so, providers=providers), "off"
    coreml = dict(COREML_OPTIONS)
    cache = "off"
    if cache_dir:
        coreml_dir = os.path.join(cache_dir, "coreml", _ort_tag(ort))
        # 어림: 폴더에 무엇이 있으면 이 모델의 변환본이 있는 것으로 본다(EP 가 맞고 틀림을 알려 주지 않는다).
        cache = "hit" if os.path.isdir(coreml_dir) and os.listdir(coreml_dir) else "miss"
        coreml["ModelCacheDirectory"] = coreml_dir
    options = [coreml if p == COREML_EP else {} for p in providers]
    return ort.InferenceSession(model_path, sess_options=so, providers=providers, provider_options=options), cache


def _new_run_options():
    import onnxruntime as ort

    return ort.RunOptions()


def _available_providers():
    import onnxruntime as ort

    return ort.get_available_providers()


def _accelerated(session):
    try:
        return any(p != CPU_EP for p in session.get_providers())
    except Exception:
        return False


def _cache_dir(ctx):
    """ctx.cache_dir(이 플러그인만 쓰는 캐시 폴더). 옛 실행기(속성 없음)나 만들 수 없으면 None."""
    return getattr(ctx, "cache_dir", None)


def _get_session(ctx, quality="precise"):
    """그 품질 모델의 캐시한 세션(없거나 품질이 바뀌었으면 만든다). 가속 장치로 못 만들면 CPU 로 만든다."""
    spec = MODELS[quality]
    model = ctx.model_path(spec["name"])
    providers = pick_providers(_available_providers(), ctx.device)
    key = (model, tuple(providers))
    if _CACHE.get("key") == key:
        return _CACHE["session"]
    _CACHE.clear()
    cache_dir = _cache_dir(ctx)
    ctx.progress(0.05, "load-model")
    start = time.monotonic()
    try:
        session, cache = _new_session(model, providers, cache_dir, ctx.log, spec=spec)
    except Exception as exc:
        if providers == [CPU_EP]:
            raise
        ctx.log(f"{providers[0]} session failed ({type(exc).__name__}: {exc}); using CPU", "warn")
        session, cache = _new_session(model, [CPU_EP], cache_dir, ctx.log, spec=spec)
    _CACHE.update(key=key, session=session)
    ctx.log(f"model loaded on {session.get_providers()} in {time.monotonic() - start:.1f}s (cache {cache})")
    return session


def _cpu_fallback(ctx, exc, quality="precise"):
    """가속 장치로 돌리다 실패했다 → CPU 세션으로 바꿔 캐시한다."""
    spec = MODELS[quality]
    key = _CACHE.get("key")
    ctx.log(f"accelerated run failed ({type(exc).__name__}: {exc}); retrying on CPU", "warn")
    _CACHE.clear()
    ctx.progress(0.2, "load-model")
    start = time.monotonic()
    session, cache = _new_session(ctx.model_path(spec["name"]), [CPU_EP], _cache_dir(ctx), ctx.log, spec=spec)
    ctx.log(f"model loaded on {session.get_providers()} in {time.monotonic() - start:.1f}s (cache {cache})")
    if key is not None:
        _CACHE.update(key=key, session=session)
    return session


def _infer(ctx, session, x, quality="precise"):
    """한 번 추론 → 출력 목록. 도는 동안 진행률을 어림해 알리고, 취소면 RunOptions.terminate 로 멈춘다."""
    feeds = {session.get_inputs()[0].name: x}
    opts = _new_run_options()
    done = threading.Event()
    expected = max(1.0, float(_STATS[quality]))
    start = time.monotonic()

    def watch():
        while not done.wait(WATCH_SEC):
            if ctx.cancelled:
                opts.terminate = True
                return
            t = time.monotonic() - start
            ctx.progress(0.2 + 0.65 * (1.0 - math.exp(-t / expected)), "infer")

    watcher = threading.Thread(target=watch, name="remove-bg-watch", daemon=True)
    ctx.progress(0.2, "infer")
    watcher.start()
    try:
        outputs = session.run(None, feeds, opts)
    except Exception:
        ctx.check_cancel()
        raise
    finally:
        done.set()
        watcher.join(2.0)
    ctx.check_cancel()
    _STATS[quality] = time.monotonic() - start
    return outputs


# ---------------------------------------------------------------------------
# 노드
# ---------------------------------------------------------------------------


def _read_image(path):
    """원본 → (RGB 배열, 원본 알파 배열 또는 None). EXIF 방향을 따른다."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.load()
        upright = ImageOps.exif_transpose(im)
    has_alpha = upright.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
        upright.mode == "P" and "transparency" in upright.info
    )
    rgba = upright.convert("RGBA")
    rgb = np.asarray(rgba.convert("RGB"), dtype=np.uint8)
    src_alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8) if has_alpha else None
    return rgb, src_alpha


def _resize_rgb(rgb, size):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def _resize_mask(mask, width, height):
    import numpy as np
    from PIL import Image

    im = Image.fromarray(np.ascontiguousarray(mask, dtype=np.float32))
    return np.asarray(im.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def remove_bg(ctx, inputs, settings):
    from PIL import Image

    edge = settings.get("edge", "soft")
    if edge not in EDGES:
        edge = "soft"
    quality, unknown = quality_of(settings)
    if unknown:
        ctx.log(f"quality {settings.get('quality')!r} is not {'/'.join(QUALITIES)}; using {quality}", "warn")
    spec = MODELS[quality]
    ctx.progress(0.02, "read")
    rgb, src_alpha = _read_image(inputs["image"])
    height, width = rgb.shape[:2]
    ctx.check_cancel()

    session = _get_session(ctx, quality)
    ctx.check_cancel()
    ctx.progress(0.18, "prepare")
    x = preprocess(_resize_rgb(rgb, SIZE), spec["mean"], spec["std"])
    try:
        outputs = _infer(ctx, session, x, quality)
    except Exception as exc:
        if ctx.cancelled or not _accelerated(session):
            raise
        session = _cpu_fallback(ctx, exc, quality)
        outputs = _infer(ctx, session, x, quality)
    del x

    ctx.progress(0.9, "compose")
    mask = _resize_mask(output_to_mask(outputs, spec), width, height)
    rgba = compose_rgba(rgb, alpha_from_mask(mask, edge), src_alpha)
    ctx.check_cancel()

    ctx.progress(0.95, "save")
    out = ctx.out_path(OUT_NAME)
    Image.fromarray(rgba).save(out, "PNG")
    ctx.progress(1.0, "done")
    return {"image": out}


NODES = {"remove-bg": remove_bg}
