import joblib
import pandas
import gradio


classifier = joblib.load("classifier.joblib")
labels = ["Falla", "Éxito"]


def predict(glasgow_intubar: int, glasgow_extubar: int, reflejo_tusigenio: str, respuesta_pupilar: str, aspiracion_secreciones: int, manejo_quirurgico: str, edad: int, sexo: str, hemoglobina: float, pafi: int, sobrecarga_hidrica: str, fiebre: str, uso_corticoide: str, dias_intubacion: int) -> dict[str, float]:
    data = {
        "Glasgow al intubar": glasgow_intubar,
        "Glasgow al extubar": glasgow_extubar,
        "Reflejo tusígenio": 0 if reflejo_tusigenio == "No" else 1,
        "Respuesta pupilar": 0 if respuesta_pupilar == "Anormal" else 1,
        "Aspiración secreciones por turno": aspiracion_secreciones,
        "Manejo quirúrgico": 0 if manejo_quirurgico == "No" else 1,
        "Edad (años)": edad,
        "Sexo": 0 if sexo == "Femenino" else 1,
        "Hemoglobina": hemoglobina,
        "PaFi": pafi,
        "Sobrecarga hídrica": 0 if sobrecarga_hidrica == "No" else 1,
        "Fiebre": 0 if fiebre == "No" else 1,
        "Uso corticoide": 0 if uso_corticoide == "No" else 1,
        "Días intubación": dias_intubacion,
    }
    x = pandas.DataFrame([data])
    y = {label: probability for label, probability in zip(labels, classifier.predict_proba(x)[0])}
    return y


interface = gradio.Interface(
    title="Traumatismo Craneoencefálico Severo (Pediátrico)",
    fn=predict,
    inputs=[
        gradio.Slider(label="Glasgow al intubar", minimum=3, maximum=15, step=1, value=9),
        gradio.Slider(label="Glasgow al extubar", minimum=3, maximum=15, step=1, value=9),
        gradio.Radio(label="Reflejo tusígenio", choices=["No", "Sí"], value="Sí"),
        gradio.Radio(label="Respuesta pupilar", choices=["Anormal", "Normal"], value="Normal"),
        gradio.Slider(label="Aspiración de secreciones por turno", minimum=0, maximum=10, step=1, value=3),
        gradio.Radio(label="Manejo quirúrgico", choices=["No", "Sí"], value="Sí"),
        gradio.Slider(label="Edad (años)", minimum=0, maximum=15, step=1, value=7),
        gradio.Radio(label="Sexo", choices=["Femenino", "Masculino"], value="Masculino"),
        gradio.Slider(label="Hemoglobina", minimum=6.0, maximum=19.0, step=0.1, value=9.8),
        gradio.Slider(label="PaFi", minimum=100, maximum=500, step=1, value=337),
        gradio.Radio(label="Sobrecarga hídrica", choices=["No", "Sí"], value="No"),
        gradio.Radio(label="Fiebre", choices=["No", "Sí"], value="No"),
        gradio.Radio(label="Uso de corticoide", choices=["No", "Sí"], value="No"),
        gradio.Slider(label="Días de intubación", minimum=0, maximum=30, step=1, value=9),
    ],
    outputs=[
        gradio.Label(label="Resultado de la extubación"),
    ],
)
interface.launch()
