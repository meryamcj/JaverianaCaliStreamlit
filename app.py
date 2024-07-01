import pandas as pd
import joblib
import streamlit as st

# Definir las columnas de entrada
input_features = ['nivsoled', 'percsalud', 'peso', 'estatura', 'imc', 'enfermedad', 'condpsiqu', 'discapacidad', 'discapvisual', 'nivactfis', 'tiemsed', 'horassuesem', 'horassuefinde', 'nivsueno', 'tociorelaj', 'tocioartist', 'tociomusic', 'tociomanual', 'tocioespirsolit', 'tocioespirgrup', 'tocioentretgrup', 'spaalcohol', 'spacigarrillo', 'spavapeo', 'spamarihuana', 'alimfrutas', 'alimverdur', 'alimembutid', 'alimpaquetes', 'alimcomidrapid', 'alimgaseos', 'alimdulces', 'alimcomidprepar', 'alimcafeteriau', 'alimcomertv', 'alimmaquinas', 'alimtiempo', 'alimhoras', 'alimcomerotros', 'alimdesayuno', 'alimrefrigmanana', 'alimalmuerzo', 'alimrefrigtarde', 'alimcena', 'alimdespucena', 'sexrelacsex', 'sexpreserv', 'sexsatisfsex', 'sexorientacsex', 'punabusotics', 'nivabusotics', 'biefisico', 'biesoc', 'bienespir', 'bieambi', 'punbienpsico', 'punbienambie', 'sexo', 'genero', 'edad', 'rangoedad', 'estratosoc', 'nivelsocioec', 'niveducativo', 'estadocivil', 'razaetnia', 'nacioen', 'nivedumadre', 'nivedupadre', 'residencia', 'residencia_valle', 'residencia_cauca', 'residencia_otro', 'zonaresidencia', 'nopersonasvivecon', 'vivepadre', 'vivemadre', 'vivehermanos', 'fpsafrontam1', 'fpsafrontam2', 'fpsafrontam3', 'fpsafrontam4', 'fpsafrontam5', 'apoyosoc', 'fpsfuncfliar1', 'punfuncfliar', 'funcfliar', 'fpsantecviol1', 'fpsantecviol2', 'fpsantecviol3', 'fpsantecviol4', 'fpsantecviol5', 'fpsantecviol6', 'cvservpub', 'cvinternet', 'cvzonasocial', 'cvcentrodepor', 'cvtransp', 'cvparques', 'cvcentrossalud', 'cvespaccomunit', 'cvsegurbarrio', 'cvviolenbarrio', 'cvinundac', 'cvruido', 'cvbasura', 'cvinvasespac', 'cvvias', 'cvtransvehicprop', 'cvtransvehicompar', 'cvtranspublicomas', 'cvtranspublicotax', 'cvtransbici', 'cvtranscamina', 'cvdecis', 'cvsust', 'cvdepecon', 'cvingresufic', 'cvingreshogar', 'cvssseps', 'cvsssmedprep', 'faculpre', 'programapre', 'semestre', 'beca', 'ica', 'nivica', 'horasclasesem', 'creditos', 'asignaturas', 'estudiatrabaja', 'horastrabaja', 'satisfacprograma', 'desempeno', 'estresores1', 'estresores2', 'estresores3', 'estresores4', 'estresores5', 'estresores6', 'estresores7', 'estresores8', 'estresores9', 'estresores10', 'estresores11', 'estresores12', 'nividpb', 'conaccprog1', 'conaccprog2', 'conaccprog3', 'conaccprog4', 'conaccprog5', 'conaccprog6', 'conaccprog7', 'conaccprog8', 'conaccprog9', 'conaccprog10', 'conaccprog11', 'conaccprog12', 'conaccprog13', 'conaccprog14', 'conaccprog15', 'conaccprog16', 'conaccprog17', 'conaccprog18', 'conaccprog19', 'ambalim1', 'ambalim2', 'ambalim3', 'ambalim4', 'ambalim5', 'ideamuerte', 'ideacsuic']

# Cargar los label_encoders
label_encoders = {
    #'feature1': joblib.load('label_encoder_feature1.joblib'),
    #'feature2': joblib.load('label_encoder_feature2.joblib'),
    # etc.
    'nivsoled': joblib.load('label_encoder_nivsoled.joblib'),
'nivresil': joblib.load('label_encoder_nivresil.joblib'),
'nivsatvida': joblib.load('label_encoder_nivsatvida.joblib'),
'nivrecpsic': joblib.load('label_encoder_nivrecpsic.joblib'),
'percsalud': joblib.load('label_encoder_percsalud.joblib'),
'imc': joblib.load('label_encoder_imc.joblib'),
'enfermedad': joblib.load('label_encoder_enfermedad.joblib'),
'condpsiqu': joblib.load('label_encoder_condpsiqu.joblib'),
'discapacidad': joblib.load('label_encoder_discapacidad.joblib'),
'discapvisual': joblib.load('label_encoder_discapvisual.joblib'),
'nivactfis': joblib.load('label_encoder_nivactfis.joblib'),
'tiemsed': joblib.load('label_encoder_tiemsed.joblib'),
'nivsueno': joblib.load('label_encoder_nivsueno.joblib'),
'tociorelaj': joblib.load('label_encoder_tociorelaj.joblib'),
'tocioartist': joblib.load('label_encoder_tocioartist.joblib'),
'tociomusic': joblib.load('label_encoder_tociomusic.joblib'),
'tociomanual': joblib.load('label_encoder_tociomanual.joblib'),
'tocioespirsolit': joblib.load('label_encoder_tocioespirsolit.joblib'),
'tocioespirgrup': joblib.load('label_encoder_tocioespirgrup.joblib'),
'tocioentretgrup': joblib.load('label_encoder_tocioentretgrup.joblib'),
'spaalcohol': joblib.load('label_encoder_spaalcohol.joblib'),
'spacigarrillo': joblib.load('label_encoder_spacigarrillo.joblib'),
'spavapeo': joblib.load('label_encoder_spavapeo.joblib'),
'spamarihuana': joblib.load('label_encoder_spamarihuana.joblib'),
'alimfrutas': joblib.load('label_encoder_alimfrutas.joblib'),
'alimverdur': joblib.load('label_encoder_alimverdur.joblib'),
'alimembutid': joblib.load('label_encoder_alimembutid.joblib'),
'alimpaquetes': joblib.load('label_encoder_alimpaquetes.joblib'),
'alimcomidrapid': joblib.load('label_encoder_alimcomidrapid.joblib'),
'alimgaseos': joblib.load('label_encoder_alimgaseos.joblib'),
'alimdulces': joblib.load('label_encoder_alimdulces.joblib'),
'alimcomidprepar': joblib.load('label_encoder_alimcomidprepar.joblib'),
'alimcafeteriau': joblib.load('label_encoder_alimcafeteriau.joblib'),
'alimcomertv': joblib.load('label_encoder_alimcomertv.joblib'),
'alimmaquinas': joblib.load('label_encoder_alimmaquinas.joblib'),
'alimtiempo': joblib.load('label_encoder_alimtiempo.joblib'),
'alimhoras': joblib.load('label_encoder_alimhoras.joblib'),
'alimcomerotros': joblib.load('label_encoder_alimcomerotros.joblib'),
'alimdesayuno': joblib.load('label_encoder_alimdesayuno.joblib'),
'alimrefrigmanana': joblib.load('label_encoder_alimrefrigmanana.joblib'),
'alimalmuerzo': joblib.load('label_encoder_alimalmuerzo.joblib'),
'alimrefrigtarde': joblib.load('label_encoder_alimrefrigtarde.joblib'),
'alimcena': joblib.load('label_encoder_alimcena.joblib'),
'alimdespucena': joblib.load('label_encoder_alimdespucena.joblib'),
'sexrelacsex': joblib.load('label_encoder_sexrelacsex.joblib'),
'sexpreserv': joblib.load('label_encoder_sexpreserv.joblib'),
'sexsatisfsex': joblib.load('label_encoder_sexsatisfsex.joblib'),
'sexorientacsex': joblib.load('label_encoder_sexorientacsex.joblib'),
'nivabusotics': joblib.load('label_encoder_nivabusotics.joblib'),
'biefisico': joblib.load('label_encoder_biefisico.joblib'),
'biesoc': joblib.load('label_encoder_biesoc.joblib'),
'bienespir': joblib.load('label_encoder_bienespir.joblib'),
'bieambi': joblib.load('label_encoder_bieambi.joblib'),
'sexo': joblib.load('label_encoder_sexo.joblib'),
'genero': joblib.load('label_encoder_genero.joblib'),
'rangoedad': joblib.load('label_encoder_rangoedad.joblib'),
'nivelsocioec': joblib.load('label_encoder_nivelsocioec.joblib'),
'niveducativo': joblib.load('label_encoder_niveducativo.joblib'),
'estadocivil': joblib.load('label_encoder_estadocivil.joblib'),
'razaetnia': joblib.load('label_encoder_razaetnia.joblib'),
'nacioen': joblib.load('label_encoder_nacioen.joblib'),
'nivedumadre': joblib.load('label_encoder_nivedumadre.joblib'),
'nivedupadre': joblib.load('label_encoder_nivedupadre.joblib'),
'residencia': joblib.load('label_encoder_residencia.joblib'),
'residencia_valle': joblib.load('label_encoder_residencia_valle.joblib'),
'residencia_cauca': joblib.load('label_encoder_residencia_cauca.joblib'),
'residencia_otro': joblib.load('label_encoder_residencia_otro.joblib'),
'zonaresidencia': joblib.load('label_encoder_zonaresidencia.joblib'),
'nopersonasvivecon': joblib.load('label_encoder_nopersonasvivecon.joblib'),
'vivepadre': joblib.load('label_encoder_vivepadre.joblib'),
'vivemadre': joblib.load('label_encoder_vivemadre.joblib'),
'vivehermanos': joblib.load('label_encoder_vivehermanos.joblib'),
'fpsafrontam1': joblib.load('label_encoder_fpsafrontam1.joblib'),
'fpsafrontam2': joblib.load('label_encoder_fpsafrontam2.joblib'),
'fpsafrontam3': joblib.load('label_encoder_fpsafrontam3.joblib'),
'fpsafrontam4': joblib.load('label_encoder_fpsafrontam4.joblib'),
'fpsafrontam5': joblib.load('label_encoder_fpsafrontam5.joblib'),
'apoyosoc': joblib.load('label_encoder_apoyosoc.joblib'),
'fpsfuncfliar1': joblib.load('label_encoder_fpsfuncfliar1.joblib'),
'funcfliar': joblib.load('label_encoder_funcfliar.joblib'),
'fpsantecviol1': joblib.load('label_encoder_fpsantecviol1.joblib'),
'fpsantecviol2': joblib.load('label_encoder_fpsantecviol2.joblib'),
'fpsantecviol3': joblib.load('label_encoder_fpsantecviol3.joblib'),
'fpsantecviol4': joblib.load('label_encoder_fpsantecviol4.joblib'),
'fpsantecviol5': joblib.load('label_encoder_fpsantecviol5.joblib'),
'fpsantecviol6': joblib.load('label_encoder_fpsantecviol6.joblib'),
'cvservpub': joblib.load('label_encoder_cvservpub.joblib'),
'cvinternet': joblib.load('label_encoder_cvinternet.joblib'),
'cvzonasocial': joblib.load('label_encoder_cvzonasocial.joblib'),
'cvcentrodepor': joblib.load('label_encoder_cvcentrodepor.joblib'),
'cvtransp': joblib.load('label_encoder_cvtransp.joblib'),
'cvparques': joblib.load('label_encoder_cvparques.joblib'),
'cvcentrossalud': joblib.load('label_encoder_cvcentrossalud.joblib'),
'cvespaccomunit': joblib.load('label_encoder_cvespaccomunit.joblib'),
'cvsegurbarrio': joblib.load('label_encoder_cvsegurbarrio.joblib'),
'cvviolenbarrio': joblib.load('label_encoder_cvviolenbarrio.joblib'),
'cvinundac': joblib.load('label_encoder_cvinundac.joblib'),
'cvruido': joblib.load('label_encoder_cvruido.joblib'),
'cvbasura': joblib.load('label_encoder_cvbasura.joblib'),
'cvinvasespac': joblib.load('label_encoder_cvinvasespac.joblib'),
'cvvias': joblib.load('label_encoder_cvvias.joblib'),
'cvtransvehicprop': joblib.load('label_encoder_cvtransvehicprop.joblib'),
'cvtransvehicompar': joblib.load('label_encoder_cvtransvehicompar.joblib'),
'cvtranspublicomas': joblib.load('label_encoder_cvtranspublicomas.joblib'),
'cvtranspublicotax': joblib.load('label_encoder_cvtranspublicotax.joblib'),
'cvtransbici': joblib.load('label_encoder_cvtransbici.joblib'),
'cvtranscamina': joblib.load('label_encoder_cvtranscamina.joblib'),
'cvdecis': joblib.load('label_encoder_cvdecis.joblib'),
'cvsust': joblib.load('label_encoder_cvsust.joblib'),
'cvdepecon': joblib.load('label_encoder_cvdepecon.joblib'),
'cvingresufic': joblib.load('label_encoder_cvingresufic.joblib'),
'cvingreshogar': joblib.load('label_encoder_cvingreshogar.joblib'),
'cvssseps': joblib.load('label_encoder_cvssseps.joblib'),
'cvsssmedprep': joblib.load('label_encoder_cvsssmedprep.joblib'),
'faculpre': joblib.load('label_encoder_faculpre.joblib'),
'programapre': joblib.load('label_encoder_programapre.joblib'),
'beca': joblib.load('label_encoder_beca.joblib'),
'nivica': joblib.load('label_encoder_nivica.joblib'),
'estudiatrabaja': joblib.load('label_encoder_estudiatrabaja.joblib'),
'satisfacprograma': joblib.load('label_encoder_satisfacprograma.joblib'),
'desempeno': joblib.load('label_encoder_desempeno.joblib'),
'estresores1': joblib.load('label_encoder_estresores1.joblib'),
'estresores2': joblib.load('label_encoder_estresores2.joblib'),
'estresores3': joblib.load('label_encoder_estresores3.joblib'),
'estresores4': joblib.load('label_encoder_estresores4.joblib'),
'estresores5': joblib.load('label_encoder_estresores5.joblib'),
'estresores6': joblib.load('label_encoder_estresores6.joblib'),
'estresores7': joblib.load('label_encoder_estresores7.joblib'),
'estresores8': joblib.load('label_encoder_estresores8.joblib'),
'estresores9': joblib.load('label_encoder_estresores9.joblib'),
'estresores10': joblib.load('label_encoder_estresores10.joblib'),
'estresores11': joblib.load('label_encoder_estresores11.joblib'),
'estresores12': joblib.load('label_encoder_estresores12.joblib'),
'nividpb': joblib.load('label_encoder_nividpb.joblib'),
'conaccprog1': joblib.load('label_encoder_conaccprog1.joblib'),
'conaccprog2': joblib.load('label_encoder_conaccprog2.joblib'),
'conaccprog3': joblib.load('label_encoder_conaccprog3.joblib'),
'conaccprog4': joblib.load('label_encoder_conaccprog4.joblib'),
'conaccprog5': joblib.load('label_encoder_conaccprog5.joblib'),
'conaccprog6': joblib.load('label_encoder_conaccprog6.joblib'),
'conaccprog7': joblib.load('label_encoder_conaccprog7.joblib'),
'conaccprog8': joblib.load('label_encoder_conaccprog8.joblib'),
'conaccprog9': joblib.load('label_encoder_conaccprog9.joblib'),
'conaccprog10': joblib.load('label_encoder_conaccprog10.joblib'),
'conaccprog11': joblib.load('label_encoder_conaccprog11.joblib'),
'conaccprog12': joblib.load('label_encoder_conaccprog12.joblib'),
'conaccprog13': joblib.load('label_encoder_conaccprog13.joblib'),
'conaccprog14': joblib.load('label_encoder_conaccprog14.joblib'),
'conaccprog15': joblib.load('label_encoder_conaccprog15.joblib'),
'conaccprog16': joblib.load('label_encoder_conaccprog16.joblib'),
'conaccprog17': joblib.load('label_encoder_conaccprog17.joblib'),
'conaccprog18': joblib.load('label_encoder_conaccprog18.joblib'),
'conaccprog19': joblib.load('label_encoder_conaccprog19.joblib'),
'ambalim1': joblib.load('label_encoder_ambalim1.joblib'),
'ambalim2': joblib.load('label_encoder_ambalim2.joblib'),
'ambalim3': joblib.load('label_encoder_ambalim3.joblib'),
'ambalim4': joblib.load('label_encoder_ambalim4.joblib'),
'ambalim5': joblib.load('label_encoder_ambalim5.joblib'),
'nivdep1': joblib.load('label_encoder_nivdep1.joblib'),
'nivans1': joblib.load('label_encoder_nivans1.joblib'),
'nivest1': joblib.load('label_encoder_nivest1.joblib'),
'ideamuerte': joblib.load('label_encoder_ideamuerte.joblib'),
'ideacsuic': joblib.load('label_encoder_ideacsuic.joblib')
}

# Streamlit app
# Cargar los modelos
models = {
    'nivrecpsic': joblib.load('tree_model_nivrecpsic.joblib'),
    'nivans1': joblib.load('tree_model_nivans1.joblib'),
    'nivdep1': joblib.load('tree_model_nivdep1.joblib'),
    'nivest1': joblib.load('tree_model_nivest1.joblib'),
    'nivsatvida': joblib.load('tree_model_nivsatvida.joblib'),
    'nivresil': joblib.load('tree_model_nivresil.joblib')
}

# Título de la aplicación
st.title("Predicción de Variables con Árboles de Decisión")

# Menú de estilo hamburguesa para seleccionar la variable objetivo
selected_variable = st.sidebar.selectbox("Seleccionar variable objetivo", list(models.keys()))

# Crear un diccionario para almacenar los datos de entrada del usuario
user_input = {}

st.header(f"Ingrese las características para predecir {selected_variable}")

for feature in input_features:
    user_input[feature] = st.text_input(f"Ingrese {feature}")

# Convertir los datos de entrada en un DataFrame
input_data = pd.DataFrame(user_input, index=[0])
def safe_transform(le, col):
    unique_labels = set(le.classes_)
    unseen_labels = set(col) - unique_labels
    if unseen_labels:
        raise ValueError(f"y contains previously unseen labels: {unseen_labels}")
    return le.transform(col)


# Realizar las transformaciones necesarias (por ejemplo, convertir las categorías a números)
for column, le in label_encoders.items():
    input_data[column] = le.transform(input_data[column])

# Categorizar los resultados predichos
def categorize_result(variable, prediction):
    categories = {
        'nivrecpsic': ['Vacío', 'Alto', 'Bajo', 'Medio'],
        'nivsatvida': ['Vacío', 'Alto', 'Bajo', 'Medio'],
        'nivresil': ['Vacío', 'Alto', 'Bajo', 'Medio'],
        'nivest1': ['Vacío', 'Leve', 'Normal', 'Moderado', 'Severo', 'Extremadamente Severo'],
        'nivdep1': ['Vacío', 'Leve', 'Normal', 'Moderado', 'Severo', 'Extremadamente Severo'],
        'nivans1': ['Vacío', 'Leve', 'Normal', 'Moderado', 'Severo', 'Extremadamente Severo']
    }
    return categories[variable][prediction]

# Hacer la predicción cuando se haga clic en el botón
if st.button("Predecir"):
    model = models[selected_variable]
    prediction = model.predict(input_data)[0]
    category = categorize_result(selected_variable, prediction)
    
    st.subheader("Resultado de la Predicción")
    st.write(f"La predicción para {selected_variable} es: {category}")
