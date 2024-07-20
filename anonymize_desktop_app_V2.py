import sys
from PyQt5.QtWidgets import QMessageBox, QListWidget,QListWidgetItem, QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLineEdit, QStatusBar, QMenuBar, QAction, QHBoxLayout
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor, QFont
from PyQt5.QtCore import Qt
from docx import Document
import fitz  # PyMuPDF
from PyQt5.QtGui import QTextDocument
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import uuid
import json
import re
import numpy as np
import os

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
from transformers import AutoConfig
from datasets import Dataset
import time
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import torch
names_matrix = np.load('ch_names.npy')

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3', 
    device='cpu',
    return_sparse=False
)

def get_embedding(text):
    with torch.cuda.amp.autocast():
        return bge_m3_ef.encode_documents([text])["dense"][0]
    
# Set TOKENIZERS_PARALLELISM to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BERT_N_TOKENS = 512
MODEL = "microsoft/Phi-3-mini-4k-instruct"
WHITELISTED_MATCHES = ["Ville de Berne", "Mon-Dossier.ch", "Berne", "Genève"]
BLACKLISTED_MATCHES = ["Nora", "Hans", "Marc", "Dubois", "Lefèvre", "Bianchi", "Zoé", "Jean", "Dufour", "Mallet", "Meier", "Elodie"]
WHITELISTED_MATCHES_EMBEDDINGS = [get_embedding(x) for x in WHITELISTED_MATCHES]
BLACKLISTED_MATCHES_EMBEDDINGS = [get_embedding(x) for x in BLACKLISTED_MATCHES]
model = AutoModelForCausalLM.from_pretrained( 
    MODEL,  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
    attn_implementation="flash_attention_2"
) 

#tokenizer = AutoTokenizer.from_pretrained("jpacifico/French-Alpaca-Phi-3-mini-4k-instruct-v1.0") 
tokenizer = AutoTokenizer.from_pretrained(MODEL) 
pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer
) 

generation_args = { 
    "max_new_tokens": 250, 
    "return_full_text": False, 
    "do_sample": False, # when set to True, it often responds with - which are annoying
}

SYSTEM_PROMPT = {
    "name": "Vous êtes un expert pour trouver des noms de famille dans du texte. Répondez simplement  par les noms de famille qui sont dans le texte. Pas de reformulations, juste les chaines de charactères qui sont dans le texte et qui correspondent à des noms de famille. UN NOM DE FAMILLE PAR LIGNE. NE DONNEZ PAS D'EXPLICATION. JUSTE LES NOMS DE FAMILLE. texte : ",
    "surname": "Vous êtes un expert pour trouver des prénoms DE PERSONNES PHYSIQUES dans du texte. Répondez simplement  par les prénoms qui sont dans le texte. Pas de reformulations, juste les chaines de charactères qui sont dans le texte et qui correspondent à des prénoms. UN prénom PAR LIGNE. NE DONNEZ PAS D'EXPLICATION. JUSTE LES PRÉNOMS. texte : ",
    "address": "Vous êtes un expert pour trouver des adresses précises dans du texte. Répondez simplement par les adresses qui sont dans le texte. Pas de reformulations, juste les chaines de charactères qui sont dans le texte et qui correspondent à des adresses. UNE ADRESSE PAR LIGNE. NE DONNEZ PAS D'EXPLICATION, JUSTE LES ADRESSES. Voici le texte : ",
    "business": "Vous êtes un expert pour trouver des noms d'entreprise dans du texte. Répondez simplement par les noms d'entreprise qui sont dans le texte. Pas de reformulations, juste les chaines de charactères qui sont dans le texte et qui correspondent à des noms d'entreprise. UN NOM D'ENTREPRISE PAR LIGNE. NE DONNEZ PAS D'EXPLICATION, JUSTE LES NOMS D'ENTREPRISE. Voici le texte : ",
    "case_number": "Vous êtes un expert pour trouver des numéro légal de dossier  dans du texte. Répondez simplement par les numéros légal de dossiers qui sont dans le texte. Pas de reformulations, juste les chaines de charactères qui sont dans le texte et qui correspondent à des des numéros légal de dossier . UN numéro de dossier légal PAR LIGNE. NE DONNEZ PAS D'EXPLICATION, JUSTE LES NUMÉROS DE LÉGAL DE DOSSIER. Voici le texte : ",
}

def get_words(text, mode):
    system_prompt = SYSTEM_PROMPT[mode]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    output = pipe(messages, **generation_args)
    words = [x.strip().replace("\n", "") for x in output[0]['generated_text'].split("\n") if x.strip() != ""]
    words = list(filter(lambda x: len(x) > 2 and x in text, words))
    return words

    
def text_to_anon_and_mapping(text_content):
    text_content = text_content.replace("\n", " ").replace("  ", " ").replace("’", "'")
    anon_mapping = []
    chunk_sizes = [400, 512]
    
    chunks_various_sizes = [
        [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
        for chunk_size in chunk_sizes
    ]
                
    w_name = set()
    w_surname = set()
    w_address = set()
    w_business = set()
    w_case_number = set()
    for chunks in chunks_various_sizes:
        print(f"Processing chunks: {chunks}")
        for t in chunks:
            w_name.update(get_words(t, "name"))
            w_surname.update(get_words(t, "surname"))
            w_address.update(get_words(t, "address"))
            w_business.update(get_words(t, "business"))
            
            words = get_words(t, "case_number")
            w_case_number.update(words)
    
    # Load the should_not_be_anon matrix
    should_not_be_anon_matrix = np.load('should_not_be_anon.npy')
    
    # Function to check if a word should be anonymized
    def should_anonymize(word):
        word_embedding = get_embedding(word)
        similarities = word_embedding @ should_not_be_anon_matrix
        max_similarity = np.max(similarities)
        if max_similarity > 0.7:
            print(f"word : {word} | max_similarity : {max_similarity}")
        return max_similarity < 0.7  # Adjust this threshold as needed
    
    excluded_w = []        
    for w in w_address:
        if should_anonymize(w):
            anon_word = "L_"+str(uuid.uuid4())[:4]
            anon_mapping.append({"word":w, "anon_word":anon_word})
            text_content = text_content.replace(w, anon_word)        
        else:
            excluded_w.append(w)    
    for w in w_name:
        if should_anonymize(w):
            anon_word = "N_"+str(uuid.uuid4())[:4]
            anon_mapping.append({"word":w, "anon_word":anon_word})
            text_content = text_content.replace(w, anon_word)
        else:
            excluded_w.append(w)    
    for w in w_surname:
        if should_anonymize(w):
            anon_word = "S_"+str(uuid.uuid4())[:4]
            anon_mapping.append({"word":w, "anon_word":anon_word})
            text_content = text_content.replace(w, anon_word)
        else:
            excluded_w.append(w)    
    for w in w_business:
        if should_anonymize(w):
            anon_word = "B_"+str(uuid.uuid4())[:4]
            anon_mapping.append({"word":w, "anon_word":anon_word})
            text_content = text_content.replace(w, anon_word)
        else:
            excluded_w.append(w)    
    for w in w_case_number:
        if should_anonymize(w):
            anon_word = "C_"+str(uuid.uuid4())[:4]
            anon_mapping.append({"word":w, "anon_word":anon_word})
            text_content = text_content.replace(w, anon_word)
        else:
            excluded_w.append(w)
            
    print(f"excluded_w : {excluded_w}")    
    print(f"len(excluded_w) : {len(excluded_w)}")    
    """
    all_words = w_name.union(w_surname).union(w_address).union(w_business).union(w_case_number)
    for w in all_words:
        for i in range(len(WHITELISTED_MATCHES_EMBEDDINGS)):
            print(f"{w} : Distance to {WHITELISTED_MATCHES[i]}: {get_embedding(w) @ (WHITELISTED_MATCHES_EMBEDDINGS[i])}")

        for i in range(len(BLACKLISTED_MATCHES_EMBEDDINGS)):
            print(f"{w} : Distance to B {BLACKLISTED_MATCHES[i]}: {get_embedding(w) @ (BLACKLISTED_MATCHES_EMBEDDINGS[i])}")
    """
    
    print("Second part : names matrix")
    
    new_words = text_content.split(" ")
    r = bge_m3_ef.encode_documents(new_words)["dense"] @ names_matrix
    names_found_counter = 0
    for i, row in enumerate(r):
        if max(row) > 0.8:
            if new_words[i] != "ne":
                print(new_words[i])
            
                text_content = text_content.replace(new_words[i], "N_"+str(uuid.uuid4())[:4])
                names_found_counter+= 1
            
    print(f"names_found_counter : {names_found_counter}")        
    return (text_content, anon_mapping)
    

class DocumentViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Document Viewer')
        self.setGeometry(100, 100, 800, 600)

        self.textEdit = QTextEdit(self)
        self.textEdit.setReadOnly(False)
        self.textEdit.setAcceptDrops(False)

        self.searchBar = QLineEdit(self)
        self.searchBar.setPlaceholderText('Search...')
        self.searchBar.textChanged.connect(self.highlight_text)

        self.searchButton = QPushButton('Search', self)
        self.searchButton.clicked.connect(self.highlight_text)

        self.openButton = QPushButton('Open File', self)
        self.openButton.clicked.connect(self.open_file)

        self.saveButton = QPushButton('Save File', self)
        self.saveButton.clicked.connect(self.save_file)

        searchLayout = QHBoxLayout()
        searchLayout.addWidget(self.searchBar)
        searchLayout.addWidget(self.searchButton)

        layout = QVBoxLayout()
        layout.addLayout(searchLayout)
        layout.addWidget(self.openButton)
        layout.addWidget(self.saveButton)
        layout.addWidget(self.textEdit)

        # New panel for anon_mapping
        self.anonListWidget = QListWidget(self)
        
        rightPanelLayout = QVBoxLayout()
        rightPanelLayout.addWidget(self.anonListWidget)

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(layout)
        mainLayout.addLayout(rightPanelLayout)

        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)

        self.menuBar = QMenuBar(self)
        self.setMenuBar(self.menuBar)

        fileMenu = self.menuBar.addMenu('File')
        openAction = QAction('Open', self)
        openAction.triggered.connect(self.open_file)
        fileMenu.addAction(openAction)

        saveAction = QAction('Save', self)
        saveAction.triggered.connect(self.save_file)
        fileMenu.addAction(saveAction)

        exitAction = QAction('Exit', self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        self.current_search_index = 0
        self.search_results = []

    def load_anon_mapping(self, anon_mapping):
        
        self.anonListWidget.clear()
        
        # Add "Names" section
        names_item = QListWidgetItem("Names & Surnames")
        names_item.setFont(QFont("Arial", weight=QFont.Bold))
        self.anonListWidget.addItem(names_item)
        
        for item in anon_mapping:
            if item['anon_word'].startswith('N_') or item['anon_word'].startswith('S_'):
                list_item = QListWidgetItem(f"{item['word']} - {item['anon_word']}")
                list_item.setFont(QFont("Arial", italic=True))
                self.anonListWidget.addItem(list_item)

        # Add "Locations" section
        locations_item = QListWidgetItem("Locations")
        locations_item.setFont(QFont("Arial", weight=QFont.Bold))
        self.anonListWidget.addItem(locations_item)
        
        for item in anon_mapping:
            if item['anon_word'].startswith('L_') or item['anon_word'].startswith('X_'):
                list_item = QListWidgetItem(f"{item['word']} - {item['anon_word']}")
                list_item.setFont(QFont("Arial", italic=True))
                self.anonListWidget.addItem(list_item)

        # Add "business" section
        business_item = QListWidgetItem("Business")
        business_item.setFont(QFont("Arial", weight=QFont.Bold))
        self.anonListWidget.addItem(business_item)
        
        for item in anon_mapping:
            if item['anon_word'].startswith('B_'):
                list_item = QListWidgetItem(f"{item['word']} - {item['anon_word']}")
                list_item.setFont(QFont("Arial", italic=True))
                self.anonListWidget.addItem(list_item)

        # Add "Case Numbers" section
        case_numbers_item = QListWidgetItem("Case Numbers")
        case_numbers_item.setFont(QFont("Arial", weight=QFont.Bold))
        self.anonListWidget.addItem(case_numbers_item)
        
        for item in anon_mapping:
            if item['anon_word'].startswith('C_'):
                list_item = QListWidgetItem(f"{item['word']} - {item['anon_word']}")
                list_item.setFont(QFont("Arial", italic=True))
                self.anonListWidget.addItem(list_item)
        
    def open_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Document", "", "Documents (*.doc *.docx *.pdf);;All Files (*)", options=options)
        if fileName:
            try:
                self.load_file(fileName)
                self.statusBar.showMessage(f"Loaded {fileName}", 5000)
            except Exception as e:
                self.statusBar.showMessage(f"Failed to load {fileName}: {str(e)}", 5000)

    def process_doc_as_text(self, text):
        anon_text, anon_mapping = text_to_anon_and_mapping(text)
        self.textEdit.setPlainText(anon_text)
        self.load_anon_mapping(anon_mapping)

    def load_file(self, file_path):
        if file_path.endswith('.doc') or file_path.endswith('.docx'):
            self.load_docx(file_path)
        elif file_path.endswith('.pdf'):
            self.load_pdf(file_path)

    def load_docx(self, file_path):
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        self.process_doc_as_text('\n'.join(full_text))
        

    def load_pdf(self, file_path):
        doc = fitz.open(file_path)
        full_text = []
        for page in doc:
            full_text.append(page.get_text())
        self.process_doc_as_text('\n'.join(full_text))

    def save_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Document", "", "Documents (*.txt);;All Files (*)", options=options)
        if fileName:
            try:
                with open(fileName, 'w') as file:
                    file.write(self.textEdit.toPlainText())
                self.statusBar.showMessage(f"Saved to {fileName}", 5000)
            except Exception as e:
                self.statusBar.showMessage(f"Failed to save {fileName}: {str(e)}", 5000)

    def highlight_text(self):
        search_text = self.searchBar.text()
        self.clear_highlight()
        self.search_results = []
        self.current_search_index = 0
        if search_text:
            format = QTextCharFormat()
            format.setBackground(QColor('yellow'))

            cursor = self.textEdit.textCursor()
            cursor.beginEditBlock()
            while self.textEdit.find(search_text, QTextDocument.FindCaseSensitively):
                self.search_results.append(cursor.selectionStart())
                cursor.mergeCharFormat(format)
            cursor.endEditBlock()

    def find_next(self):
        if self.search_results:
            self.current_search_index = (self.current_search_index + 1) % len(self.search_results)
            cursor = self.textEdit.textCursor()
            cursor.setPosition(self.search_results[self.current_search_index])
            self.textEdit.setTextCursor(cursor)

    def clear_highlight(self):
        cursor = self.textEdit.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.setCharFormat(QTextCharFormat())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            try:
                self.load_file(file_path)
                self.statusBar.showMessage(f"Loaded {file_path}", 5000)
            except Exception as e:
                self.statusBar.showMessage(f"Failed to load {file_path}: {str(e)}", 5000)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DocumentViewer()
    viewer.setAcceptDrops(True)
    viewer.show()
    sys.exit(app.exec_())