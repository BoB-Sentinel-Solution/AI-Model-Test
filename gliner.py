# gliner_ko_tiny10.py
# -----------------------------------------------------------
# Test 10 hard-coded prompts on taeminlee/gliner_ko (GLiNER-kr)
# - Print-only (no files)
# - Zero-shot labels (custom ontology)
# - (옵션) --with_regex 로 패턴형 민감정보 보강
#
# Setup:
#   pip install -U gliner torch transformers
#
# Run:
#   python gliner_ko_tiny10.py
#   python gliner_ko_tiny10.py --threshold 0.35 --with_regex
# -----------------------------------------------------------

import argparse, json, re, time
from typing import List, Dict, Any

# 1) 테스트 프롬프트 10개 (하드코딩)
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

# 2) GLiNER 라벨(타입) 목록 + 간단 설명(모델에 힌트 제공)
LABELS = [
    # 사람/연락/주소/주문/계정
    "PERSON_NAME(사람 이름)", "PHONE_NUMBER(전화번호)", "EMAIL(이메일)", "ADDRESS(주소)",
    "ORDER_ID(주문번호)", "ACCOUNT_ID(계정 ID)", "BROWSER(브라우저 이름)",
    # 인증/보안 토큰
    "PASSWORD(비밀번호)", "TOTP_BACKUP_CODE(2FA 백업코드)", "IP_ADDRESS(IP 주소)",
    "API_KEY(API 키)", "GITHUB_PAT(GitHub 개인 액세스 토큰)", "SLACK_TOKEN(Slack 토큰)",
    "JWT(JSON Web Token)", "REFRESH_TOKEN(리프레시 토큰)",
    # 결제/계좌
    "CARD_NUMBER(신용카드 번호)", "CARD_EXPIRY(카드 유효기간)", "CARD_CVV(카드 보안코드)", "CARD_HOLDER(카드 소유자)",
    "IBAN(국제계좌번호)", "BIC(은행 식별코드)", "BANK_NAME(은행명)", "BRANCH_NAME(지점명)", "ACCOUNT_NUMBER(계좌번호)",
    "CURRENCY(통화)", "AMOUNT(금액)",
    # 비즈니스 식별자
    "CUSTOMER_ID(고객번호)", "CRM_ID(CRM 레코드 ID)", "INVOICE_ID(송장번호)", "COUPON_CODE(쿠폰 코드)", "GATEWAY_CUSTOMER_ID(게이트웨이 고객ID)",
    # 세션/쿠키
    "COOKIE(쿠키)", "CSRF_TOKEN(CSRF 토큰)", "SESSION_ID(세션 ID)",
]

# 3) (옵션) 정규식 보강 - 패턴형 민감정보
REGEXES = [
    ("PHONE_NUMBER", re.compile(r"\b0\d{1,2}-\d{3,4}-\d{4}\b")),
    ("EMAIL",        re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")),
    ("IP_ADDRESS",   re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("API_KEY",      re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("GITHUB_PAT",   re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")),
    ("SLACK_TOKEN",  re.compile(r"\bxox[bap]-[A-Za-z0-9-]{10,}\b")),
    ("JWT",          re.compile(r"\beyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b")),
    ("CARD_NUMBER",  re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("CARD_EXPIRY",  re.compile(r"\b(?:0[1-9]|1[0-2])/(?:\d{2}|\d{4})\b")),
    ("CARD_CVV",     re.compile(r"\b\d{3,4}\b")),
    ("IBAN",         re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("BIC",          re.compile(r"\b[A-Z0-9]{8}(?:[A-Z0-9]{3})?\b")),
    ("COOKIE",       re.compile(r"\b(sessionid|csrftoken|XSRF-TOKEN|XSRF|CSRF)[=\:][^\s;]+\b", re.IGNORECASE)),
]

def regex_extract(prompt: str) -> List[Dict[str, str]]:
    out, seen = [], set()
    for t, pat in REGEXES:
        for m in pat.finditer(prompt):
            v = m.group(0)
            k = (t, v, m.start(), m.end())
            if k in seen:
                continue
            seen.add(k)
            out.append({"type": t, "value": v})
    return out

def canonicalize(label: str) -> str:
    # GLiNER 라벨에서 괄호/설명 제거 + 대문자/언더스코어 통일
    base = label.split("(")[0].strip()
    return base.upper().replace(" ", "_")

def dedup(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    res, seen = [], set()
    for e in entities:
        t = e.get("type", "").strip()
        v = e.get("value", "").strip()
        if not t or not v:
            continue
        key = (t, v)
        if key in seen:
            continue
        seen.add(key)
        res.append({"type": t, "value": v})
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="taeminlee/gliner_ko", help="GLiNER Korean model id")
    ap.add_argument("--threshold", type=float, default=0.35, help="GLiNER confidence threshold")
    ap.add_argument("--with_regex", action="store_true", help="정규식 보강 사용")
    args = ap.parse_args()

    # GLiNER 로드
    try:
        from gliner import GLiNER
    except ImportError:
        print("`gliner` 패키지를 찾을 수 없습니다. 먼저 설치하세요: pip install gliner")
        return

    print(f"[INFO] Loading GLiNER model: {args.model}")
    t0 = time.perf_counter()
    model = GLiNER.from_pretrained(args.model)
    print(f"[INFO] Loaded in {time.perf_counter() - t0:.2f}s")

    labels = LABELS  # 필요 시 커스터마이즈 가능

    # 한 개씩 테스트(간단한 시간 측정)
    for i, p in enumerate(PROMPTS, 1):
        t1 = time.perf_counter()
        ents = model.predict_entities(p, labels, threshold=args.threshold)
        # GLiNER 결과 -> {type, value}
        gliner_entities = [{"type": canonicalize(e.get("label", "")),
                            "value": e.get("text", "")} for e in ents]

        if args.with_regex:
            gliner_entities += regex_extract(p)

        final_entities = dedup(gliner_entities)
        has_sensitive = bool(final_entities)

        result = {"has_sensitive": has_sensitive, "entities": final_entities}
        elap_ms = (time.perf_counter() - t1) * 1000.0

        print(f"\n--- TEST #{i} ---")
        print("prompt:", p)
        print(f"latency_ms: {elap_ms:.2f}")
        print("output:", json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
