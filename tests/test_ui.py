from streamlit.testing.v1 import AppTest

from src.config import ROOT


def test_streamlit_demo():
    app = AppTest.from_file(str(ROOT / 'src/app/streamlit_app.py')).run(timeout=20)
    assert not app.exception
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert len(app.metric) == 3
    assert app.metric[2].value == '24/24'
    assert any('Синтетическое' in item.value for item in app.info)
