import json
import time
import argparse
import requests

def normalize_label(x: str) -> str:
    return (x or "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="เช่น https://...run.app/predict")
    ap.add_argument("--file", default="test.jsonl", help="ไฟล์ข้อมูลทดสอบแบบ JSONL")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    total = 0
    passed = 0
    failed_ids = []
    errors = []

    with open(args.file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            item = json.loads(line)

            news_id = item.get("id")
            headline = item.get("topic", "")  # topic -> headline
            body = item.get("body", "")
            true_label = normalize_label(item.get("label"))

            payload = {"headline": headline, "body": body}

            t0 = time.time()
            try:
                r = requests.post(args.url, json=payload, timeout=args.timeout)
                latency_ms = round((time.time() - t0) * 1000, 2)

                if r.status_code != 200:
                    errors.append({
                        "id": news_id,
                        "line": line_no,
                        "status": r.status_code,
                        "latency_ms": latency_ms,
                        "body": r.text[:300]
                    })
                    continue

                data = r.json()
                pred_label = normalize_label(data.get("label"))  # API ของคุณคืน key = "label"
                ok = (pred_label == true_label)

                if ok:
                    passed += 1
                else:
                    failed_ids.append({
                        "id": news_id,
                        "true": true_label,
                        "pred": pred_label,
                        "confidence": data.get("confidence"),
                        "latency_ms": data.get("latency_ms", latency_ms),
                    })

            except Exception as e:
                errors.append({
                    "id": news_id,
                    "line": line_no,
                    "error": str(e)
                })

    # สรุปผล
    print("=== TEST SUMMARY ===")
    print(f"Total:  {total}")
    print(f"Pass:   {passed}")
    print(f"Fail:   {len(failed_ids)}")
    print(f"Errors: {len(errors)}")

    if failed_ids:
        print("\n=== FAILED IDS ===")
        for x in failed_ids:
            print(f"- id={x['id']} true={x['true']} pred={x['pred']} conf={x['confidence']} latency={x['latency_ms']}ms")

    if errors:
        print("\n=== ERRORS ===")
        for e in errors[:20]:
            print(f"- id={e.get('id')} line={e.get('line')} status={e.get('status')} err={e.get('error')} body={e.get('body')}")
        if len(errors) > 20:
            print(f"... and {len(errors)-20} more errors")

    # ถ้าอยากให้ CI fail เมื่อมี fail/errors:
    if failed_ids or errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
