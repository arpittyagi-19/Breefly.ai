from flask import Flask, render_template, request
from utils.extractor import extract_text_from_youtube, extract_text_from_pdf, extract_text_from_txt
from utils.rag_pipeline import build_rag_qa_chain

app = Flask(__name__)

qa_chain = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global qa_chain

    if request.method == 'POST':
        upload_type = request.form['source_type']
        if upload_type == 'youtube':
            video_url = request.form['youtube_url']
            text = extract_text_from_youtube(video_url)
        elif upload_type == 'pdf':
            file = request.files['pdf_file']
            text = extract_text_from_pdf(file)
        elif upload_type == 'txt':
            file = request.files['txt_file']
            text = extract_text_from_txt(file)

        if not text:
            return render_template('index.html', error="No content found!")

        qa_chain = build_rag_qa_chain(text)
        return render_template('answer.html')

    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    global qa_chain
    question = request.form['question']
    answer = qa_chain(question) if qa_chain else "Please upload a document first."
    return render_template('answer.html', question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
