import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

# Windows 控制台默认是 GBK，日志/output 里的 ✓ 等符号会抛 UnicodeEncodeError。
# 统一按 UTF-8 输出，无法编码的字符替换掉而不是崩。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# 抑制 OTel 因生成器/上下文交互产生的 "Failed to detach context" 警告。
# 链路追踪不受影响。
# 已知缺陷：https://github.com/open-telemetry/opentelemetry-python/issues/2606
class _SuppressOtelDetachWarning(logging.Filter):
    def filter(self, record):
        return "Failed to detach context" not in record.getMessage()

logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachWarning())

from ui.css import custom_css
from ui.gradio_app import create_gradio_ui

if __name__ == "__main__":
    print("\n🔨 Creating RAG Assistant...")
    demo = create_gradio_ui()
    print("\n🚀 Launching RAG Assistant...")
    demo.launch(css=custom_css)