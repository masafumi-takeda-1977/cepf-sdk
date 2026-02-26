# apps/run_pipeline.py
import queue
import threading
import traceback
import time

from cepf_sdk.airy import UdpAiryDecoder, AiryDecodeConfig
from apps.processor import processor_loop, ProcessorConfig

def producer_loop(q: "queue.Queue"):
    try:
        print("[producer] start", flush=True)
        cfg = AiryDecodeConfig(port=6699, agg_seconds=0.1)
        print(f"[producer] cfg.port={cfg.port} agg={cfg.agg_seconds}", flush=True)

        dec = UdpAiryDecoder(cfg)
        print("[producer] decoder created; entering frames()", flush=True)

        for k, frame in enumerate(dec.frames(), start=1):
            if k == 1:
                print(f"[producer] got first frame: id={frame.metadata.frame_id} points={frame.point_count}", flush=True)
            if k % 10 == 0:
                print(f"[producer] frames={k} last_points={frame.point_count}", flush=True)

            q.put(frame)
    except Exception:
        print("[producer] EXCEPTION:\n" + traceback.format_exc(), flush=True)
        raise

def consumer_loop(q: "queue.Queue"):
    try:
        print("[consumer] start", flush=True)
        processor_loop(
            q,
            config=ProcessorConfig(print_every=10, start_from=1, point_index=10000),
        )
    except Exception:
        print("[consumer] EXCEPTION:\n" + traceback.format_exc(), flush=True)
        raise

def main():
    q: "queue.Queue" = queue.Queue(maxsize=10)

    th_p = threading.Thread(target=producer_loop, args=(q,), daemon=False)
    th_c = threading.Thread(target=consumer_loop, args=(q,), daemon=False)

    th_p.start()
    th_c.start()

    while True:
        time.sleep(2.0)
        print(f"[main] alive qsize={q.qsize()}", flush=True)

if __name__ == "__main__":
    main()
