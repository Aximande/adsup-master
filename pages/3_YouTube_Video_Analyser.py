import streamlit as st
import yt_dlp
from pydub import AudioSegment
from openai import OpenAI
import os
from langcodes import Language
import glob
import base64
#from utils import load_custom_css

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# French translations
TRANSLATIONS = {
    "YouTube Video Transcriber and Analyzer": "Transcripteur et Analyseur de Vidéos YouTube",
    "Enter YouTube Video URL": "Entrez l'URL de la vidéo YouTube",
    "Select the language spoken in the video": "Sélectionnez la langue parlée dans la vidéo",
    "Enter specific terms or acronyms (one per line)": "Entrez des termes ou acronymes spécifiques (un par ligne)",
    "Select the language for the analysis": "Sélectionnez la langue pour l'analyse",
    "Select the level of detail for the final report": "Sélectionnez le niveau de détail pour le rapport final",
    "Brief": "Bref",
    "Moderate": "Modéré",
    "Detailed": "Détaillé",
    "Process Video": "Traiter la Vidéo",
    "Downloading audio from YouTube...": "Téléchargement de l'audio depuis YouTube...",
    "Audio downloaded successfully.": "Audio téléchargé avec succès.",
    "Downloaded audio size: {:.2f} MB": "Taille de l'audio téléchargé : {:.2f} Mo",
    "Error downloading audio.": "Erreur lors du téléchargement de l'audio.",
    "Error downloading video:": "Erreur lors du téléchargement de la vidéo :",
    "Error loading audio file:": "Erreur lors du chargement du fichier audio :",
    "Audio duration: {:.2f} minutes": "Durée de l'audio : {:.2f} minutes",
    "Estimated Whisper transcription cost: ${:.4f}": "Coût estimé de la transcription Whisper : {:.4f} €",
    "Transcribing chunk {}/{}...": "Transcription du segment {}/{}...",
    "Error during transcription:": "Erreur lors de la transcription :",
    "Post-processing transcript with GPT-4o mini...": "Post-traitement de la transcription avec GPT-4o mini...",
    "GPT-4o mini usage: {} input tokens, {} output tokens": "Utilisation de GPT-4o mini : {} tokens d'entrée, {} tokens de sortie",
    "Estimated GPT-4o mini cost for post-processing: ${:.4f}": "Coût estimé de GPT-4o mini pour le post-traitement : {:.4f} €",
    "Error during post-processing:": "Erreur lors du post-traitement :",
    "Analyzing transcript with GPT-4o mini...": "Analyse de la transcription avec GPT-4o mini...",
    "Estimated GPT-4o mini cost for analysis: ${:.4f}": "Coût estimé de GPT-4o mini pour l'analyse : {:.4f} €",
    "Error during analysis:": "Erreur lors de l'analyse :",
    "Detailed Analysis": "Analyse Détaillée",
    "Total Estimated Cost": "Coût Total Estimé",
    "Total estimated cost for all operations: ${:.4f}": "Coût total estimé pour toutes les opérations : {:.4f} €",
    "Download Report": "Télécharger le Rapport",
    "Audio file is larger than 25 MB, splitting into chunks...": "Le fichier audio dépasse 25 Mo, découpage en segments...",
    "The corrected transcript is too short or empty. Unable to proceed with analysis.": "La transcription corrigée est trop courte ou vide. Impossible de procéder à l'analyse.",
    "Processing chunk {}/{}...": "Traitement du segment {}/{}...",
    "Error: Corrected transcript too short or empty": "Erreur : la transcription corrigée est trop courte ou vide",
    "Selected detail level is invalid.": "Le niveau de détail sélectionné est invalide.",
    "Error: Unexpected transcription format": "Erreur : format de transcription inattendu",
    "Full report saved as {}": "Rapport complet enregistré sous {}",
}

# Cost constants
WHISPER_COST_PER_MINUTE = 0.006
GPT4O_MINI_INPUT_COST = 0.150 / 1_000_000  # $0.150 per million input tokens
GPT4O_MINI_OUTPUT_COST = 0.600 / 1_000_000  # $0.600 per million output tokens

# Charger le CSS personnalisé
#load_custom_css()


def translate(text, *args):
    """Translate text to French. If formatting is needed, args can be provided."""
    translated = TRANSLATIONS.get(text, text)  # Fallback to original text if no translation
    if args:
        return translated.format(*args)
    return translated

@st.cache_data
def generate_download_link(text, filename, link_text):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'

@st.cache_data
def get_video_title(video_url):
    ydl_opts = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return info.get('title', 'Untitled Video')

@st.cache_data
def download_audio(video_url):
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'outtmpl': 'audio.%(ext)s',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([video_url])
    return error_code

def cleanup_temp_files():
    for file in glob.glob("audio_chunk_*.mp3"):
        os.remove(file)
    if os.path.exists("audio.m4a"):
        os.remove("audio.m4a")
    if os.path.exists(report_filename):
        os.remove(report_filename)

@st.cache_data
def process_audio_chunk(chunk_file, language_code):
    with open(chunk_file, "rb") as audio_file_obj:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_obj,
            language=language_code,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return response

@st.cache_data
def analyze_transcript_with_gpt(corrected_transcript, analysis_prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specializing in content analysis."},
            {"role": "user", "content": analysis_prompt}
        ],
        max_tokens=16384
    )
    return response

def split_audio(audio, max_chunk_size_mb=10):
    chunk_length_ms = len(audio) * (max_chunk_size_mb / (len(audio) / 1000 / 60 * 1.5))  # Estimation plus précise
    chunks = []
    for i, start in enumerate(range(0, len(audio), int(chunk_length_ms))):
        chunk = audio[start:start + int(chunk_length_ms)]
        chunk_file = f"audio_chunk_{i}.mp3"
        chunk.export(chunk_file, format="mp3")
        chunks.append(chunk_file)
    return chunks

def process_chunks(chunks, language_code):
    transcripts = []
    for idx, chunk_file in enumerate(chunks):
        st.write(translate("Processing chunk {}/{}...", idx + 1, len(chunks)))
        transcript = process_audio_chunk(chunk_file, language_code)
        transcripts.append(transcript.text)
    return " ".join(transcripts)

def chunk_text(text, max_tokens=4000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word.split()) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word.split())
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def analyze_large_transcript(transcript, analysis_prompt):
    chunks = chunk_text(transcript)
    analyses = []
    for idx, chunk in enumerate(chunks):
        st.write(f"Analyzing chunk {idx + 1}/{len(chunks)}...")
        chunk_prompt = f"{analysis_prompt}\n\nTranscript chunk {idx + 1}/{len(chunks)}:\n{chunk}"
        analysis = analyze_transcript_with_gpt(chunk, chunk_prompt)
        analyses.append(analysis.choices[0].message.content)
    return "\n\n".join(analyses)

st.title(translate("YouTube Video Transcriber and Analyzer"))

video_url = st.text_input(translate("Enter YouTube Video URL"))

languages = [
    'English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese', 'Russian',
    'Arabic', 'Portuguese', 'Hindi', 'Bengali', 'Urdu', 'Indonesian', 'Other'
]
selected_language = st.selectbox(translate("Select the language spoken in the video"), languages)

terms_input = st.text_area(translate("Enter specific terms or acronyms (one per line)"))

analysis_language = st.selectbox(translate("Select the language for the analysis"), languages)

# Detail level options with internal keys and display labels
DETAIL_LEVEL_OPTIONS = [
    {'key': 'Brief', 'label': translate('Brief')},
    {'key': 'Moderate', 'label': translate('Moderate')},
    {'key': 'Detailed', 'label': translate('Detailed')}
]

detail_level_display = st.select_slider(
    translate("Select the level of detail for the final report"),
    options=[option['label'] for option in DETAIL_LEVEL_OPTIONS],
    value=translate("Moderate")
)

# Map back from display label to internal key
detail_level = next(
    (option['key'] for option in DETAIL_LEVEL_OPTIONS if option['label'] == detail_level_display),
    None
)

if detail_level is None:
    st.error(translate("Selected detail level is invalid."))
    st.stop()

if st.button(translate("Process Video")):
    total_cost = 0
    process_log = []  # To store all process information

    if video_url:
        try:
            st.info(translate("Downloading audio from YouTube..."))
            error_code = download_audio(video_url)
            if error_code == 0:
                st.success(translate("Audio downloaded successfully."))
                file_size = os.path.getsize('audio.m4a') / (1024 * 1024)  # in MB
                st.info(translate("Downloaded audio size: {:.2f} MB", file_size))
                process_log.append(f"Audio downloaded successfully. Size: {file_size:.2f} MB")
            else:
                st.error(translate("Error downloading audio."))
                process_log.append("Error downloading audio.")
                st.stop()
        except Exception as e:
            st.error(f"{translate('Error downloading video:')} {str(e)}")
            process_log.append(f"Error downloading video: {str(e)}")
            st.stop()

    audio_file = 'audio.m4a'
    try:
        audio = AudioSegment.from_file(audio_file)
        audio_duration_minutes = len(audio) / 60000  # Convert milliseconds to minutes
        whisper_cost = audio_duration_minutes * WHISPER_COST_PER_MINUTE
        total_cost += whisper_cost
        st.info(translate("Audio duration: {:.2f} minutes", audio_duration_minutes))
        st.info(translate("Estimated Whisper transcription cost: ${:.4f}", whisper_cost))
        process_log.append(f"Audio duration: {audio_duration_minutes:.2f} minutes")
        process_log.append(f"Estimated Whisper transcription cost: ${whisper_cost:.4f}")
    except Exception as e:
        st.error(f"{translate('Error loading audio file:')} {str(e)}")
        process_log.append(f"Error loading audio file: {str(e)}")
        st.stop()

    language_code = Language.find(selected_language).language

    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    max_size_mb = 25

    if file_size_mb > max_size_mb:
        st.info(translate("Audio file is larger than 25 MB, splitting into chunks..."))
        chunk_length_ms = len(audio) * (max_size_mb / file_size_mb)  # Estimate chunk length
        chunks = []
        for i, start in enumerate(range(0, len(audio), int(chunk_length_ms))):
            chunk = audio[start:start + int(chunk_length_ms)]
            chunk_file = f"audio_chunk_{i}.mp3"
            chunk.export(chunk_file, format="mp3")
            chunks.append(chunk_file)
        process_log.append(f"Audio split into {len(chunks)} chunks")
    else:
        chunks = [audio_file]
        process_log.append("Audio processed as a single file")

    transcripts = []
    for idx, chunk_file in enumerate(chunks):
        st.write(translate("Processing chunk {}/{}...", idx + 1, len(chunks)))
        try:
            transcript = process_audio_chunk(chunk_file, language_code)
            transcripts.append(transcript)
            process_log.append(f"Chunk {idx + 1}/{len(chunks)} transcribed successfully")
        except Exception as e:
            st.error(f"{translate('Error during transcription:')} {str(e)}")
            process_log.append(f"Error transcribing chunk {idx + 1}/{len(chunks)}: {str(e)}")
            st.stop()

    # Combine transcripts
    combined_transcript = ""
    for transcript in transcripts:
        if hasattr(transcript, 'text'):
            combined_transcript += transcript.text + " "
        elif isinstance(transcript, dict) and 'text' in transcript:
            combined_transcript += transcript['text'] + " "
        else:
            st.error(translate("Error: Unexpected transcription format"))
            st.write(transcript)
            process_log.append("Error: Unexpected transcription format")
            st.stop()

    process_log.append("Transcripts combined successfully")

    # Post-process transcript with GPT-4o mini
    terms_list = terms_input.strip().split('\n')
    terms_string = ', '.join(terms_list)

    system_prompt = f"""
You are a helpful assistant. Correct any misspellings in the transcript.
Ensure that the following terms are spelled correctly: {terms_string}.
Only make necessary corrections and do not change other parts of the text.
"""

    # Since 'gpt-4o-mini' has a large context length, we can process the transcript in one go
    st.info(translate("Post-processing transcript with GPT-4o mini..."))
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_transcript}
            ],
            max_tokens=16384  # Adjust as needed
        )
        corrected_transcript = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        gpt_cost = (input_tokens * GPT4O_MINI_INPUT_COST) + (output_tokens * GPT4O_MINI_OUTPUT_COST)
        total_cost += gpt_cost
        st.info(translate("GPT-4o mini usage: {} input tokens, {} output tokens", input_tokens, output_tokens))
        st.info(translate("Estimated GPT-4o mini cost for post-processing: ${:.4f}", gpt_cost))
        process_log.append(f"Transcript post-processed. Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        process_log.append(f"Post-processing cost: ${gpt_cost:.4f}")
    except Exception as e:
        st.error(f"{translate('Error during post-processing:')} {str(e)}")
        process_log.append(f"Error during post-processing: {str(e)}")
        st.stop()

    # Check if the corrected transcript is too short
    if len(corrected_transcript.strip()) < 50:  # Adjust this threshold as needed
        st.error(translate("The corrected transcript is too short or empty. Unable to proceed with analysis."))
        process_log.append("Error: Corrected transcript too short or empty")
        st.stop()

    # Analyze transcript with GPT-4o mini
    detail_instructions = {
        "Brief": "Provide a concise overview focusing on the main topics and overall message. Include up to 3 key points with their respective timestamps.",
        "Moderate": "Offer a balanced summary with specific points and some examples. Include up to 5 key points with their respective timestamps and brief explanations.",
        "Detailed": "Deliver an in-depth analysis with extensive examples and subpoints. Include all main chapters and sub-chapters key points with their respective timestamps, detailed explanations, and relevant examples or quotes from the transcript. Ensure that you give equal attention to all parts of the transcript, including the middle and end sections."
    }

    instruction = detail_instructions[detail_level]

    analysis_prompt = f"""
Analyze the following transcript and provide a {detail_level.lower()} summary in {analysis_language}.
{instruction}
Organize the analysis in a clear, structured format, using markdown for formatting.
Begin with an overview of the content, then list the main points or chapters chronologically.
For each main point or chapter, include the timestamp, a brief description, and any relevant details or examples.
Ensure that you give equal attention to all parts of the transcript, including the middle and end sections.
Conclude with a summary of the overall message or significance of the content.

Transcript:
{{TRANSCRIPT_CHUNK}}
"""

    st.info(translate("Analyzing transcript with GPT-4o mini..."))
    try:
        analysis_text = analyze_large_transcript(corrected_transcript, analysis_prompt)
        process_log.append("Transcript analyzed successfully")
    except Exception as e:
        st.error(f"{translate('Error during analysis:')} {str(e)}")
        process_log.append(f"Error during analysis: {str(e)}")
        st.stop()

    st.header(translate("Detailed Analysis"))
    st.markdown(analysis_text)

    st.header(translate("Total Estimated Cost"))
    st.info(translate("Total estimated cost for all operations: ${:.4f}", total_cost))
    process_log.append(f"Total estimated cost: ${total_cost:.4f}")

    video_title = get_video_title(video_url)
    safe_title = ''.join(c for c in video_title if c.isalnum() or c in (' ', '_', '-'))[:50]  # Limit length and remove special characters
    report_filename = f"{safe_title}_analysis.txt"

    # Prepare the full report content
    full_report = f"""Video Title: {video_title}

Detailed Analysis:
{analysis_text}

Original Transcript:
{combined_transcript}

Corrected Transcript:
{corrected_transcript}

Total Estimated Cost: ${total_cost:.4f}
"""

    # Libération de la mémoire
    del audio
    del chunks
    del transcripts
    del combined_transcript

    # Nettoyage des fichiers temporaires
    cleanup_temp_files()

    # Génération du lien de téléchargement
    download_link = generate_download_link(full_report, report_filename, translate("Download Report"))
    st.markdown(download_link, unsafe_allow_html=True)

    # Save the full report locally
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(full_report)
    st.success(translate("Full report saved as {}", report_filename))

# Ajoutez ce code à la fin de votre script principal :
if st.button("Clear Session"):
    st.cache_data.clear()
    cleanup_temp_files()
    st.experimental_rerun()
