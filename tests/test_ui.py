from streamlit.testing.v1 import AppTest

from src.config import ROOT


def test_streamlit_demo():
    app = AppTest.from_file(str(ROOT / 'src/app/streamlit_app.py')).run(timeout=30)
    app.sidebar.radio[0].set_value('Синтетическое демо').run()
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 3
    assert app.metric[2].value == '24/24'
    assert any('Синтетическое' in item.value for item in app.info)


def test_streamlit_real_data_default():
    app = AppTest.from_file(str(ROOT / 'src/app/streamlit_app.py')).run(timeout=30)
    assert app.sidebar.radio[0].value == 'Реальные данные Kelmarsh'
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 6
    assert app.metric[5].value == '144/144'
    assert any('MODE A' in item.value for item in app.info)
