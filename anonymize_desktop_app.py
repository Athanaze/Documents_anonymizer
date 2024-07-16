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
BERT_N_TOKENS = 512
from llama_cpp import Llama
llm = Llama(
  model_path="./Phi-3-mini-4k-instruct-q4.gguf", 
  n_ctx=4096,  # No change in perf observed by just changing that and keeping the same input prompt
  n_threads=4, # 1-4 to perf increase, after 4 perf decrease
  n_gpu_layers=999, # faster with 0... needs more testing
  temperature=0
)
prompt = """
Put all the addresses that you can find in the attached text in a JSON CODE BLOCK LIKE SO ```json []``` ok? respond with ONLY THE JSON CODE BLOCK. here is the attachment:
 """
prompt = """
Mettez toutes les adresses que vous pouvez trouver dans le texte joint dans un BLOC DE CODE JSON COMME CECI ```json [{"adresse":"adresse 1"}, {"adresse":"adresse 2"}, ...]```. Répondez avec SEULEMENT LE BLOC DE CODE JSON. voici la pièce jointe:
 """
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

#tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
#model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")
CITIES = ["Genève", "Crans-Montana", "Lausanne", "Yverdon-les-Bains", "Montreux", "Nyon", "Vevey", "Renens", "Pully", "Morges", "Aigle", "Prilly", "Fribourg", "Bulle", "Villars-sur-Glâne", "Marly", "Givisiez", "Granges-Paccot", "Tafers", "Düdingen", "Murten", "Kerzers"]
OTHERS_TO_EXCLUDE = ["avocat", "monsieur", "madame", "des", "du", "les"]
    
def text_to_anon_and_mapping(text_content):
    
    anon_mapping = []
    lines = text_content.split("\n")

    def tokenize_line(line):
        tokens = tokenizer(line, return_tensors="pt", truncation=True, padding=True)
        return tokens["input_ids"][0].tolist()

    def split_string_into_n_chunks(s, n):
        k, m = divmod(len(s), n)
        return [s[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    def get_words_from_ner_results(ner_results, of_type):
        l = list(filter(lambda x: x["entity"] in of_type, ner_results))
        sticked_to_next = [False]*len(l)

        for i, x in enumerate(l):
            if(i+1 < len(l) and l[i+1]["start"] == l[i]["end"]):
                sticked_to_next[i] = True

        c = ""
        words = []
        for i in range(len(l)):
            w_to_add = l[i]["word"].replace("##", "")
            if(sticked_to_next[i]):
                c+=w_to_add
            else:
                c+=w_to_add
                if(len(c)>2):
                    words.append(c)
                
                c = ""
        return words
        
    chunks = []

    for line in lines:
        if(line != ""):
            t = tokenize_line(line)
            if(len(t) > BERT_N_TOKENS):
                splitting_number = len(t) / BERT_N_TOKENS
                for chunk in split_string_into_n_chunks(line, splitting_number+1):
                    chunks.append(chunk)
                
            else:
                chunks.append(line)
            
    person_words_set = set()
    org_words_set = set()
    for c in chunks:
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0)
        ner_results = nlp(c)
        person_words = get_words_from_ner_results(ner_results, ["B-PER", "I-PER"])
        #org_words = get_words_from_ner_results(ner_results, ["B-ORG", "I-ORG"])
        org_words = get_words_from_ner_results(ner_results, ["WILL NOT MATCH"])
        person_words_set.update(list(filter(lambda x: x not in CITIES and x not in OTHERS_TO_EXCLUDE, person_words)))
        org_words_set.update(list(filter(lambda x: x not in CITIES and x.lower() not in OTHERS_TO_EXCLUDE, org_words)))

    org_words_set.difference_update(person_words_set)
    
    for w in person_words_set:
        anon_mapping.append({"word":w, "anon_word":"N_"+str(uuid.uuid4())[:4]})
        
    for o in org_words_set:
        anon_mapping.append({"word":o, "anon_word":"L_"+str(uuid.uuid4())[:4]})  

    
    for w in anon_mapping:
        text_content = text_content.replace(w["word"], w["anon_word"])
    
    
    # text_split_lines = text_content.split("\n")
    
    # for l in text_split_lines:
    #     # Simple inference example
    #     output = llm(
    #     f"<|user|>\n{prompt} {l}<|end|>\n<|assistant|>",
    #     max_tokens=512,  # Generate up to 256 tokens
    #     stop=["<|end|>"], 
    #     echo=False,  # Whether to echo the prompt
    #     )
    #     print(l)
    #     print(output['choices'][0]['text'])
    output = llm(
    f"<|user|>\n{prompt} {text_content}<|end|>\n<|assistant|>",
    max_tokens=512,  # Generate up to 256 tokens
    stop=["<|end|>"], 
    echo=False,  # Whether to echo the prompt
    )
    # Create a QMessageBox to display the output
    
    llm_response = output['choices'][0]['text']
    
    msg_box = QMessageBox()
    msg_box.setWindowTitle("LLM Output")
    msg_box.setText(llm_response)
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec_()
    
    print(llm_response)
    # Find the JSON content within the ```json [...] ``` block
    json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
    if json_match:
        json_content = json_match.group(1)
        try:
            address_list = json.loads(json_content)
            
            # Replace addresses in text_content
            for address_dict in address_list:
                if 'adresse' in address_dict:
                    address_to_redact = address_dict['adresse']
                    text_content = text_content.replace(address_to_redact, '[REDACTED ADDRESS]')
        
        except json.JSONDecodeError:
            print("Error decoding JSON from LLM response")
    else:
        print("No JSON content found in LLM response")
    
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
        names_item = QListWidgetItem("Names")
        names_item.setFont(QFont("Arial", weight=QFont.Bold))
        self.anonListWidget.addItem(names_item)
        
        for item in anon_mapping:
            if item['anon_word'].startswith('N_'):
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