# koelectra_ner_regex_tiny10.py
# -----------------------------------------------------------
# pip install -U transformers torch
# Run: python koelectra_ner_regex_tiny10.py
#   (NER 끄고 정규식만으로 빠르게 보기: python ... --regex_only)
# -----------------------------------------------------------

import argparse, json, re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

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

# --- 정규식(보안/식별자) ---
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
]

# --- NER 라벨 → 통일 타입 맵(대략적): NAVER/KLUE 관례 ---
MAP_NER = {
    "PS": "PERSON_NAME", "PERSON": "PERSON_NAME",
    "LC": "LOCATION", "L": "LOCATION", "ADDR": "ADDRESS",
    "OG": "ORGANIZATION", "ORG": "ORGANIZATION",
}

def run_regex(prompt: str) -> List[Dict]:
    found = []
    taken = set()
    for t, pat in REGEXES:
        for m in pat.finditer(prompt):
            v = m.group(0)
            key = (t, v, m.start(), m.end())
            if key in taken: 
                continue
            taken.add(key)
            found.append({"type": t, "value": v})
    return found

def run_ner(prompt: str, nlp=None) -> List[Dict]:
    if nlp is None: 
        return []
    out = nlp(prompt)
    ents = []
    for e in out:
        # e example: {'entity_group': 'PS', 'word': '김민서', 'start': 10, 'end': 13, 'score': 0.99}
        tag = e.get("entity_group") or e.get("entity") or ""
        tag = tag.replace("B-","").replace("I-","").upper()
        t = MAP_NER.get(tag)
        if not t: 
            continue
        ents.append({"type": t, "value": e["word"]})
    return ents

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ner_model", default="monologg/koelectra-base-v3-naver-ner")
    ap.add_argument("--regex_only", action="store_true", help="정규식만 사용 (NER 비활성)")
    args = ap.parse_args()

    nlp = None
    if not args.regex_only:
        tok = AutoTokenizer.from_pretrained(args.ner_model, trust_remote_code=True)
        mdl = AutoModelForTokenClassification.from_pretrained(args.ner_model, trust_remote_code=True)
        nlp = pipeline("token-classification", model=mdl, tokenizer=tok,
                       aggregation_strategy="simple", device=-1)  # CPU

    for i, p in enumerate(PROMPTS, 1):
        ents = run_regex(p)
        ents += run_ner(p, nlp)

        # 중복 제거(타입+값 기준)
        uniq = []
        seen = set()
        for e in ents:
            k = (e["type"], e["value"])
            if k in seen: 
                continue
            seen.add(k)
            uniq.append(e)

        has_sensitive = bool(uniq)
        j = {"has_sensitive": has_sensitive, "entities": uniq}

        print(f"\n--- TEST #{i} ---")
        print("prompt:", p)
        print("output:", json.dumps(j, ensure_ascii=False))

if __name__ == "__main__":
    main()
