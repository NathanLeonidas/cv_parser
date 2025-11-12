from pdfminer.high_level import extract_text
import re
import unicodedata
import re

def remove_vaporwave_spaces(txt):
    text = ' ' + txt + ' '
    # Découper le texte en tokens (mots et espaces) tout en conservant les espaces.
    clear = ''
    i=1
    while i<len(text)-1:
        count=0
        j = i
        temp=''
        if text[i]!=' ' and text[i-1]==' 'and text[i+1]==' ':
            while j<len(text)-1 and text[j]!=' ' and text[j-1]==' 'and text[j+1]==' ':
                temp+=text[j]
                j+=2
                count+=1
            if count>=3:
                clear+=temp
                i = j + 1
            else:
                clear+=text[i]
                i+=1
        else:
            clear+=text[i]
            i+=1
    return re.sub(r'\s+', ' ', clear).strip()



def extract_txt_preprocessed(path):
    # Fusionner les lettres séparées par des espaces
    text = extract_text(path)
    return text

def process(txt):
# Normalisation : mise en minuscule
    txt = txt.lower()

    txt = re.sub(r"[\'’‘‛ʼꞌʹ:]", ' ', txt)
    txt = unicodedata.normalize("NFD", txt).encode("ascii", "ignore").decode("utf-8")
    txt = re.sub(r'\s+', ' ', txt.replace("\n", " ").replace("\r", " ")).strip()
    txt = re.sub(r'[^\w\s+@/\\.-]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    txt = remove_vaporwave_spaces(txt)
    return txt
