# qwen25_tiny10.py  — Qwen 2.5 "7B" 버전
# -----------------------------------------------------------
# pip install -U transformers accelerate torch sentencepiece
# 예) python qwen25_tiny10.py --int4
# -----------------------------------------------------------

import argparse, json, time, threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers.utils import is_flash_attn_2_available

SYS_PROMPT = (
    """
    You are a strict detector for sensitive entities (PII and secrets).

    Return ONLY a compact JSON with these keys:
    - has_sensitive: true or false
    - entities: list of {"type": <LABEL>, "value": <exact substring>}

    HARD RULES
    - Allowed labels ONLY (uppercase, exact match). If a label is not in the list below, DO NOT invent or output it.
    - If the text contains none of the allowed entities: return exactly {"has_sensitive": false, "entities": []}.
    - `value` must be the exact substring from the user text (no masking, no redaction, no normalization).
    - Output JSON only — no explanations, no extra text, no code fences, no trailing commas.
    - The JSON must be valid and parseable.

    ALLOWED LABELS
    # 1) Basic Identity Information
    NAME, PHONE, EMAIL, ADDRESS, POSTAL_CODE,
  
    # 2) Public Identification Number
    PERSONAL_CUSTOMS_ID, RESIDENT_ID, PASSPORT, DRIVER_LICENSE, FOREIGNER_ID, HEALTH_INSURANCE_ID, BUSINESS_IDMILITARY_ID,

    # 3) Authentication Information
    JWT, API_KEY, GITHUB_PAT, PRIVATE_KEY,

    # 4) Finanacial Information
    CARD_NUMBER, CARD_EXPIRY, BANK_ACCOUNT, CARD_CVV, PAYMENT_PIN, MOBILE_PAYMENT_PIN, PAYMENT_URI_QR,

    # 5) Cryptocurrency Information
    MNEMONIC, CRYPTO_PRIVATE_KEY, HD_WALLET,

    # 6) Network Information + etc
    IPV4, IPV6, MAC_ADDRESS, IMEI
    """
    )

# 하드코딩된 10개 프롬프트
PROMPTS = [
    "주문 확인 좀 해줄래요? 주문ID ORDER-25-0912-ABCD-7788, 수령인 김민서, 연락처 010-1234-5678, 이메일 minseo.kim@example.com, 주소 서울 강남구 테헤란로 12 5층(06234) 입니다.",
    "계정 잠금 해제 가능한가요? 계정ID ms.park, 비밀번호 SuperSecret!234, 2FA 백업코드 ABCD-EFGH-IJKL-MNOP, 최근 로그인 IP 192.168.10.45, 브라우저 Chrome 입니다.",
    "이 메시지에 민감정보가 있나요? OpenAI API 키 sk-EXAMPLEFAKEKEY1234567890abcd, GitHub PAT ghp_EXAMPLEExampleExampleExample1234, Slack 토큰 xoxb-12345-EXAMPLE-abcdefghijkl 이 포함돼요.",
    "아래 JWT와 refresh_token이 유효한지 점검해줄 수 있나요? Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.EXAMPLE.zzZEXAMPLE / refresh_token=eyJraWQiOiJLT0RFIiJ9.EXAMPLE.SIGN",
    "해외 송금 정보는 어떻게 표시되나요? IBAN DE89370400440532013000, BIC DEUTDEFF, 수취인 이름 Hans Müller, 금액 EUR 1,250.00 입니다.",
    "카드 결제 실패 원인 알려줄래요? 카드번호 4111 1111 1111 1111, 유효기간 12/27, CVV 123, 카드소유자 Choi Hana, 주문ID ORDER-2025-HELLO-9876 입니다.",
    "계좌이체 내역 표시 기준이 궁금해요. 은행 국민은행 강남지점, 예금주 홍길동, 계좌번호 123-456-789012, 통화 KRW, 금액 ₩1,980,000 입니다.",
    "고객·주문 식별자 마스킹 예시 보여줄래요? 고객번호 CUST-002931, CRM 레코드ID CRM-7F2A-11EE-BC12, 주문ID ORDER-2025-0912-XYZ-7788, 연락처 010-2233-4455 입니다.",
    "청구 관련 문의입니다. 송장번호 INV-2025-000123, 청구지 부산 해운대구 A로 77 1203호, 게이트웨이 고객ID cus_FAKE12345, 쿠폰 CP-TEST-8899 가 포함돼요.",
    "세션/쿠키가 포함된 로그는 어떻게 처리돼요? Cookie: sessionid=s%3AEXAMPLE._SIG_; XSRF-TOKEN=EXAMPLETOKEN123; csrftoken=xyz123"
]

def build_chat(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_json_from_text(s: str):
    try:
        start = s.index("{"); end = s.rindex("}")
    except ValueError:
        return None
    block = s[start:end+1]
    try:
        return json.loads(block)
    except Exception:
        block2 = block.replace("True","true").replace("False","false")
        try:
            return json.loads(block2)
        except Exception:
            return None

def load_model(model_id: str, int4: bool):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    if int4:
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb,
                dtype=torch.float16,
                attn_implementation=attn_impl,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print("[WARN] 4bit 실패 → BF16/FP32:", e)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                attn_implementation=attn_impl,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            attn_implementation=attn_impl,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return tok, model

@torch.inference_mode()
def infer_once(tokenizer, model, prompt: str, max_new_tokens=96) -> dict:
    text = build_chat(tokenizer, prompt)
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )

    t0 = time.perf_counter()
    out_text, first_token_time = [], None

    def _consume():
        nonlocal first_token_time
        for chunk in streamer:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            out_text.append(chunk)

    th = threading.Thread(target=_consume); th.start()
    _ = model.generate(**gen_kwargs)
    th.join()
    t1 = time.perf_counter()

    gen_text = "".join(out_text).strip()
    ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else None
    gen_tokens = len(tokenizer.encode(gen_text, add_special_tokens=False))
    gen_time_s = (t1 - (first_token_time or t1)) if first_token_time else 0.0
    tok_per_s = (gen_tokens / gen_time_s) if gen_time_s > 0 else None

    parsed = parse_json_from_text(gen_text)
    return {
        "raw_output": gen_text, "json": parsed,
        "ttft_ms": ttft_ms, "tok_per_s": tok_per_s,
    }

def main():
    ap = argparse.ArgumentParser()
    # ★ 기본값을 7B로 변경
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--int4", action="store_true", help="4bit 양자화 로딩")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()

    print(f"[INFO] Loading: {args.model} (int4={args.int4})")
    tokenizer, model = load_model(args.model, args.int4)

    for i, p in enumerate(PROMPTS, 1):
        res = infer_once(tokenizer, model, p, max_new_tokens=args.max_new_tokens)
        j = res["json"]
        print(f"\n--- TEST #{i} ---")
        print("prompt:", p)
        print("ttft_ms:", f"{res['ttft_ms']:.2f}" if res["ttft_ms"] else "NA",
              "| tok/s:", f"{res['tok_per_s']:.2f}" if res["ttft_ms"] and res["tok_per_s"] else "NA")
        print("output:", res["raw_output"] if res["raw_output"] else "")
        print("parsed_json:", json.dumps(j, ensure_ascii=False) if j is not None else "None")

if __name__ == "__main__":
    main()
