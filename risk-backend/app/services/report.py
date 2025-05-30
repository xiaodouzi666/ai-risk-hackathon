from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import io, json, pathlib

env = Environment(loader=FileSystemLoader(pathlib.Path(__file__).parent / "templates"))

def build_pdf(data: dict, model_url: str) -> bytes:
    html = env.get_template("report.html").render(
        model_url=model_url,
        results=data
    )
    return HTML(string=html).write_pdf()


def build_json(data: dict) -> bytes:
    return json.dumps(data, indent=2).encode()
