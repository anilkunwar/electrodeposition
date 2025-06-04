import streamlit as st
import PyPDF2
import tempfile
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from itertools import combinations
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import logging
import seaborn as sns
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
import spacy
from math import log
import uuid
import json
import pandas as pd
import yaml

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize session state
if 'custom_stopwords' not in st.session_state:
    st.session_state.custom_stopwords = "et al,figure,table"
if 'clear_selections' not in st.session_state:
    st.session_state.clear_selections = False
if 'custom_idf' not in st.session_state:
    st.session_state.custom_idf = {}

# Download NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK data already present.")
        return True
    except LookupError:
        logger.info("Downloading NLTK punkt_tab and stopwords...")
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {str(e)}")
        st.error(f"Failed to download NLTK data: {str(e)}")
        return False

if not download_nltk_data():
    st.stop()

# Cache spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy en_core_web_sm model.")
        return nlp
    except OSError:
        logger.info("Downloading spaCy en_core_web_sm model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        logger.info("Downloaded and loaded spaCy model.")
        return nlp

nlp = load_spacy_model()

# Default keywords in YAML format for Phase Field PINN Cu Template Electrodeposition
DEFAULT_KEYWORDS_YAML = """
categories:
  phase_field_models:
    name: Phase Field Models
    keywords:
      - Allen-Cahn equation
      - Cahn-Hilliard equation
      - Ginzburg-Landau functional
      - phase-field modeling
      - spinodal decomposition
      - dendritic solidification
      - order parameter
      - interface dynamics
      - double-well potential
      - gradient flow
  physics_informed_neural_networks:
    name: Physics-Informed Neural Networks (PINNs)
    keywords:
      - physics-informed neural networks
      - PINNs
      - neural network architecture
      - causal training
      - Fourier features
      - random Fourier features
      - loss function
      - automatic differentiation
      - Adam optimizer
      - spectral bias
  electrodeposition_processes:
    name: Electrodeposition Processes
    keywords:
      - electrodeposition
      - dendrite formation
      - overpotential
      - Butler-Volmer equation
      - ion flux
      - chemical potential
      - lithium-ion transport
      - electrochemical reaction
      - metal plating
      - concentration-driven diffusion
      - pulsed electrodeposition
      - ionic liquid electrolyte
      - aluminum nano-rods
      - current collector
      - template-assisted deposition
      - Sb-tartrate electrolyte
      - pulsed-current deposition
      - nano-pillar coating
      - Cu nano-rods
      - homogeneous coverage
      - electrolyte decomposition
  numerical_methods:
    name: Numerical Methods
    keywords:
      - finite element method
      - FEM
      - finite difference method
      - numerical solver
      - computational mesh
      - adaptive error control
      - MOOSE framework
      - FEniCS
      - computational cost
      - linear system
  battery_technologies:
    name: Battery Technologies
    keywords:
      - lithium-ion batteries
      - metal electrodes
      - sodium electrodes
      - zinc electrodes
      - anode
      - cathode
      - electrolyte
      - redox reaction
      - Coulombic efficiency
      - short-circuiting
      - 3D micro-batteries
      - intercalation
      - rocking-chair battery
      - anatase TiO₂
      - lithium alloying
      - lithium-alloying anode
      - gravimetric capacity
      - capacity retention
      - cycling stability
      - Sn-based anode
      - LiCoO₂ cathode
  material_parameters:
    name: Material Parameters
    keywords:
      - interfacial mobility
      - surface tension
      - interface thickness
      - diffusivity
      - conductivity
      - molar fraction
      - charge-transfer coefficient
      - applied potential
      - site density
      - barrier height
      - area gain
      - electrochemical stability window
      - viscosity
      - adherence
      - volume expansion
      - particle size
      - morphology
      - Li⁺ transference number
      - electrolyte uptake
      - mechanical integrity
  3d_micro_battery_architectures:
    name: 3D Micro-Battery Architectures
    keywords:
      - interdigitated arrays
      - interpenetrated electrodes
      - aperiodic sponge-like configuration
      - nano-structured current collector
      - conformal electrolyte layer
      - Cu₂Sb electrode
      - nano-structured anode
      - hybrid polymer separator
      - spray coating
      - trenched structure
"""

# Function to load keywords from YAML
@st.cache_data
def load_keywords(yaml_content: str) -> dict:
    try:
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            raise ValueError("YAML content must be a dictionary")
        if 'categories' in data:
            keywords = {cat: data['categories'][cat]['keywords'] for cat in data['categories']}
        else:
            keywords = data
        for category, terms in keywords.items():
            if not isinstance(terms, list):
                raise ValueError(f"Category '{category}' must contain a list of keywords")
            keywords[category] = [str(term).lower() for term in terms]
        logger.info("Loaded keywords from YAML")
        return keywords
    except Exception as e:
        logger.error(f"Error parsing YAML content: {str(e)}")
        return None

# Load IDF_APPROX
try:
    json_path = os.path.join(os.path.dirname(__file__), "idf_approx.json")
    with open(json_path, "r") as f:
        IDF_APPROX = json.load(f)
    logger.info("Loaded IDF_APPROX from idf_approx.json")
except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
    logger.warning(f"Failed to load idf_approx.json from {json_path}: {str(e)}. Using default IDF_APPROX.")
    st.warning(f"Could not load idf_approx.json: {str(e)}. Falling back to hardcoded IDF values.")
    IDF_APPROX = {
        "electrodeposition": log(1000 / 100),
        "pulsed electrodeposition": log(1000 / 50),
        "phase-field modeling": log(1000 / 50),
        "physics-informed neural networks": log(1000 / 50),
        "lithium-ion batteries": log(1000 / 100),
        "3D micro-batteries": log(1000 / 50),
        "volume expansion": log(1000 / 50),
        "Cu₂Sb electrode": log(1000 / 10),
        "hybrid polymer separator": log(1000 / 10)
    }
DEFAULT_IDF = log(100000 / 10000)

PHYSICS_CATEGORIES = ["material_parameters", "3d_micro_battery_architectures"]

# Visualization options
COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Blues", "Greens", "Oranges", "Reds", "YlOrBr", "YlOrRd",
    "PuBu", "BuPu", "GnBu", "PuRd", "RdPu",
    "coolwarm", "Spectral", "PiYG", "PRGn", "RdYlBu",
    "twilight", "hsv", "tab10", "Set1", "Set2", "Set3"
]
NETWORK_STYLES = ["seaborn-v0_8-white", "ggplot", "bmh", "classic", "dark_background"]
NODE_SHAPES = ['o', 's', '^', 'v', '>', '<', 'd', 'p', 'h']
EDGE_STYLES = ['solid', 'dashed', 'dotted', 'dashdot']
COLORS = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'gray', 'white']
FONT_FAMILIES = ['Arial', 'Helvetica', 'Times New Roman', 'Courier New', 'Verdana']
BBOX_COLORS = ['black', 'white', 'gray', 'lightgray', 'lightblue', 'lightyellow']
LAYOUT_ALGORITHMS = ['spring', 'circular', 'kamada_kawai', 'shell', 'spectral', 'random', 'spiral', 'planar']
WORD_ORIENTATIONS = ['horizontal', 'vertical', 'random']

# Initialize session state for custom stopwords
if 'custom_stopwords' not in st.session_state:
    st.session_state.custom_stopwords = "et al,figure,table"

# Existing functions (with fixes for NameError)
def estimate_idf(term, word_freq, total_words, idf_approx, keyword_categories, nlp_model):
    if 'custom_idf' not in st.session_state:
        st.session_state.custom_idf = {}
    if term in st.session_state.custom_idf:
        logger.debug(f"Using cached IDF for {term}: {st.session_state.custom_idf[term]}")
        return st.session_state.custom_idf[term]
    tf = word_freq.get(term, 1) / total_words
    freq_idf = log(1 / max(tf, 1e-6))
    freq_idf = min(freq_idf, 8.517)
    sim_idf = DEFAULT_IDF
    max_similarity = 0.0
    term_doc = nlp_model(term)
    for known_term in idf_approx:
        known_doc = nlp_model(known_term)
        similarity = term_doc.similarity(known_doc)
        if similarity > max_similarity and similarity > 0.7:
            max_similarity = similarity
            sim_idf = idf_approx[known_term]
            logger.debug(f"Similarity match for {term}: {known_term} (sim={similarity:.2f}, IDF={sim_idf:.3f})")
    cat_idf = DEFAULT_IDF
    for category, keywords in keyword_categories.items():
        if any(k in term or term in k for k in keywords):
            cat_idfs = [idf_approx.get(k, DEFAULT_IDF) for k in keywords if k in idf_approx]
            if cat_idfs:
                cat_idf = sum(cat_idfs) / len(cat_idfs)
                logger.debug(f"Category match for {term}: {category} (avg IDF={cat_idf:.3f})")
                break
    if max_similarity > 0.7:
        estimated_idf = 0.7 * sim_idf + 0.2 * freq_idf + 0.1 * cat_idf
    else:
        estimated_idf = 0.4 * freq_idf + 0.4 * cat_idf + 0.2 * DEFAULT_IDF
    estimated_idf = max(2.303, min(8.517, estimated_idf))
    st.session_state.custom_idf[term] = estimated_idf
    logger.debug(f"Estimated IDF for {term}: {estimated_idf:.3f} (freq={freq_idf:.3f}, sim={sim_idf:.3f}, cat={cat_idf:.3f})")
    return estimated_idf

def get_candidate_keywords(text, min_freq, min_length, use_stopwords, custom_stopwords, exclude_keywords, top_limit, tfidf_weight, use_nouns_only, include_phrases):
    stop_words = set(stopwords.words('english')) if use_stopwords else set()
    stop_words.update(['introduction', 'conclusion', 'section', 'chapter', 'the', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
    stop_words.update([w.strip().lower() for w in custom_stopwords.split(",") if w.strip()])
    exclude_set = set([w.strip().lower() for w in exclude_keywords.split(",") if w.strip()])
    words = word_tokenize(text.lower())
    if use_nouns_only:
        doc = nlp(text)
        nouns = {token.text.lower() for token in doc if token.pos_ == "NOUN"}
        filtered_words = [w for w in words if w in nouns and w.isalnum() and len(w) >= min_length and w not in stop_words and w not in exclude_set]
    else:
        filtered_words = [w for w in words if w.isalnum() and len(w) >= min_length and w not in stop_words and w not in exclude_set]
    word_freq = Counter(filtered_words)
    logger.debug("Single word frequencies: %s", word_freq.most_common(20))
    phrases = []
    if include_phrases:
        doc = nlp(text)
        raw_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1 and len(chunk.text) >= min_length]
        phrases = [clean_phrase(phrase, stop_words) for phrase in raw_phrases if clean_phrase(phrase, stop_words)]
        phrases = [p for p in phrases if p not in stop_words and p not in exclude_set]
        phrase_freq = Counter(phrases)
        phrases = [(p, f) for p, f in phrase_freq.items() if f >= min_freq]
        logger.debug("Extracted phrases: %s", phrases[:20])
    total_words = len(word_tokenize(text))
    tfidf_scores = {}
    idf_sources = {}
    for word, freq in word_freq.items():
        if freq < min_freq:
            continue
        tf = freq / total_words
        if word in IDF_APPROX:
            idf = IDF_APPROX[word]
            source = "JSON"
        else:
            idf = estimate_idf(word, word_freq, total_words, IDF_APPROX, KEYWORD_CATEGORIES, nlp)
            source = "Estimated"
        tfidf_scores[word] = tf * idf * tfidf_weight
        idf_sources[word] = {"idf": idf, "source": source, "frequency": freq}
        logger.debug(f"PDF term {word}: TF-IDF={tfidf_scores[word]:.3f}, IDF={idf:.3f}, Source={source}, Freq={freq}")
    for phrase, freq in phrases:
        if freq < min_freq:
            continue
        tf = freq / total_words
        if phrase in IDF_APPROX:
            idf = IDF_APPROX[phrase]
            source = "JSON"
        else:
            idf = estimate_idf(phrase, phrase_freq, total_words, IDF_APPROX, KEYWORD_CATEGORIES, nlp)
            source = "Estimated"
        tfidf_scores[phrase] = tf * idf * tfidf_weight
        idf_sources[phrase] = {"idf": idf, "source": source, "frequency": freq}
        logger.debug(f"PDF term {phrase}: TF-IDF={tfidf_scores[phrase]:.3f}, IDF={idf:.3f}, Source={source}, Freq={freq}")
    for term in tfidf_scores:
        for category, keywords in KEYWORD_CATEGORIES.items():
            if term in keywords and category in PHYSICS_CATEGORIES:
                tfidf_scores[term] *= 1.5
                logger.debug(f"Boosted TF-IDF for {term}: {tfidf_scores[term]:.3f}")
    if tfidf_weight > 0:
        ranked_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_limit]
    else:
        ranked_terms = [(w, f) for w, f in word_freq.most_common(top_limit) if f >= min_freq]
        ranked_terms += phrases[:top_limit - len(ranked_terms)]
    categorized_keywords = {cat: [] for cat in KEYWORD_CATEGORIES}
    term_to_category = {}
    for term, score in ranked_terms:
        if term in exclude_set:
            continue
        assigned = False
        for category, keywords in KEYWORD_CATEGORIES.items():
            if term in keywords:
                categorized_keywords[category].append((term, score))
                term_to_category[term] = category
                assigned = True
                break
            elif " " in term:
                if any(k == term or term.startswith(k + " ") or term.endswith(" " + k) for k in keywords):
                    categorized_keywords[category].append((term, score))
                    term_to_category[term] = category
                    assigned = True
                    break
        if not assigned:
            categorized_keywords["material_parameters"].append((term, score))
            term_to_category[term] = "material_parameters"
    logger.debug("Categorized keywords: %s", {k: [t[0] for t in v] for k, v in categorized_keywords.items()})
    return categorized_keywords, word_freq, phrases, tfidf_scores, term_to_category, idf_sources

@st.cache_data
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name
        pdf_reader = PyPDF2.PdfReader(tmp_file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        os.unlink(tmp_file_path)
        logger.info("Extracted text from PDF")
        return text if text.strip() else "No text extracted from the PDF."
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return f"Error extracting text: {str(e)}"

def extract_text_between_phrases(text, start_phrase, end_phrase):
    try:
        start_idx = text.find(start_phrase)
        end_idx = text.find(end_phrase, start_idx + len(start_phrase))
        if start_idx == -1 or end_idx == -1:
            return "Specified phrases not found in the text."
        return text[start_idx:end_idx + len(end_phrase)]
    except Exception as e:
        logger.error(f"Error extracting text between phrases: {str(e)}")
        return f"Error extracting text between phrases: {str(e)}"

def clean_phrase(phrase, stop_words):
    words = phrase.split()
    while words and words[0].lower() in stop_words:
        words = words[1:]
    while words and words[-1].lower() in stop_words:
        words = words[:-1]
    return " ".join(words).strip()

def generate_word_cloud(
    text, selected_keywords, tfidf_scores, selection_criteria, colormap,
    title_font_size, caption_font_size, font_step, word_orientation, background_color,
    contour_width, contour_color
):
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['laser', 'microstructure', 'the', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
        stop_words.update([w.strip().lower() for w in st.session_state.custom_stopwords.split(",") if w.strip()])
        processed_text = text.lower()
        keyword_map = {}
        for keyword in selected_keywords:
            internal_key = keyword.replace(" ", "_")
            processed_text = processed_text.replace(keyword, internal_key)
            keyword_map[internal_key] = keyword
        words = word_tokenize(processed_text)
        filtered_words = [keyword_map.get(word, word) for word in words if keyword_map.get(word, word) in selected_keywords]
        if not filtered_words:
            return None, "No valid words or phrases found for word cloud after filtering."
        frequencies = {word: tfidf_scores.get(word, 1.0) for word in filtered_words}
        max_freq = max(frequencies.values(), default=1.0)
        frequencies = {word: freq / max_freq for word, freq in frequencies.items()}
        wordcloud = WordCloud(
            width=1600, height=800,
            background_color=background_color,
            min_font_size=8,
            max_font_size=200,
            font_step=font_step,
            prefer_horizontal=1.0 if word_orientation == 'horizontal' else 0.0 if word_orientation == 'vertical' else 0.5,
            colormap=colormap,
            contour_width=contour_width,
            contour_color=contour_color,
            margin=10
        ).generate_from_frequencies(frequencies)
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Cloud of Phase Field PINN Template Electrodeposition Keywords", fontsize=title_font_size, pad=20, fontweight='bold')
        caption = f"Word Cloud generated with: {selection_criteria}"
        plt.figtext(
            0.5, 0.02, caption, ha="center", fontsize=caption_font_size, wrap=True,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        return fig, None
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, f"Error generating word cloud: {str(e)}"

def generate_bibliometric_network(
    text, selected_keywords, tfidf_scores, label_font_size, selection_criteria,
    node_colormap, edge_colormap, network_style, line_thickness, node_alpha, edge_alpha,
    title_font_size, caption_font_size, node_size_scale, node_shape, node_linewidth,
    node_edgecolor, edge_style, label_font_color, label_font_family, label_bbox_facecolor,
    label_bbox_alpha, layout_algorithm, label_rotation, label_offset
):
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['laser', 'microstructure', 'the', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
        stop_words.update([w.strip().lower() for w in st.session_state.custom_stopwords.split(",") if w.strip()])
        processed_text = text.lower()
        keyword_map = {}
        for keyword in selected_keywords:
            internal_key = keyword.replace(" ", "_")
            processed_text = processed_text.replace(keyword, internal_key)
            keyword_map[internal_key] = keyword
        words = word_tokenize(processed_text)
        filtered_words = [keyword_map.get(word, word) for word in words if keyword_map.get(word, word) in selected_keywords]
        word_freq = Counter(filtered_words)
        if not word_freq:
            return None, "No valid words or phrases found for bibliometric network."
        top_words = [word for word, freq in word_freq.most_common(20)]
        sentences = sent_tokenize(text.lower())
        co_occurrences = Counter()
        for sentence in sentences:
            processed_sentence = sentence
            for keyword in selected_keywords:
                processed_sentence = processed_sentence.replace(keyword, keyword.replace(" ", "_"))
            words_in_sentence = [keyword_map.get(word, word) for word in word_tokenize(processed_sentence) if keyword_map.get(word, word) in top_words]
            for pair in combinations(set(words_in_sentence), 2):
                co_occurrences[tuple(sorted(pair))] += 1
        G = nx.Graph()
        for word, freq in word_freq.most_common(20):
            G.add_node(word, size=freq)
        for (word1, word2), weight in co_occurrences.items():
            if word1 in top_words and word2 in top_words:
                G.add_edge(word1, word2, weight=weight)
        communities = greedy_modularity_communities(G)
        node_colors = {}
        try:
            cmap = plt.cm.get_cmap(node_colormap)
            palette = cmap(np.linspace(0.2, 0.8, max(1, len(communities))))
        except ValueError:
            logger.warning(f"Invalid node colormap {node_colormap}, falling back to viridis")
            cmap = plt.cm.get_cmap("viridis")
            palette = cmap(np.linspace(0.2, 0.8, max(1, len(communities))))
        for i, community in enumerate(communities):
            for node in community:
                node_colors[node] = palette[i]
        edge_weights = [G.edges[edge]['weight'] for edge in G.edges]
        max_weight = max(edge_weights, default=1)
        edge_widths = [line_thickness * (1 + 2 * np.log1p(weight / max_weight)) for weight in edge_weights]
        try:
            edge_cmap = plt.cm.get_cmap(edge_colormap)
            edge_colors = [edge_cmap(weight / max_weight) for weight in edge_weights]
        except ValueError:
            logger.warning(f"Invalid edge colormap {edge_colormap}, falling back to Blues")
            edge_cmap = plt.cm.get_cmap("Blues")
            edge_colors = [edge_cmap(weight / max_weight) for weight in edge_weights]
        try:
            if layout_algorithm == 'spring':
                pos = nx.spring_layout(G, k=0.8, seed=42)
            elif layout_algorithm == 'circular':
                pos = nx.circular_layout(G, scale=1.2)
            elif layout_algorithm == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            elif layout_algorithm == 'shell':
                pos = nx.shell_layout(G)
            elif layout_algorithm == 'spectral':
                pos = nx.spectral_layout(G)
            elif layout_algorithm == 'random':
                pos = nx.random_layout(G, seed=42)
            elif layout_algorithm == 'spiral':
                pos = nx.spiral_layout(G)
            elif layout_algorithm == 'planar':
                try:
                    pos = nx.planar_layout(G)
                except nx.NetworkXException:
                    logger.warning("Graph is not planar, falling back to spring layout")
                    pos = nx.spring_layout(G, k=0.8, seed=42)
        except Exception as e:
            logger.error(f"Error in layout {layout_algorithm}: {str(e)}, falling back to spring")
            pos = nx.spring_layout(G, k=0.8, seed=42)
        try:
            plt.style.use(network_style)
        except ValueError:
            logger.warning(f"Invalid network style {network_style}, falling back to seaborn-v0_8-white")
            plt.style.use("seaborn-v0_8-white")
        plt.rcParams['font.family'] = label_font_family
        fig, ax = plt.subplots(figsize=(16, 12), dpi=400)
        node_sizes = [G.nodes[node]['size'] * node_size_scale * 20 for node in G.nodes]
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=[node_colors[node] for node in G.nodes],
            node_shape=node_shape,
            edgecolors=node_edgecolor,
            linewidths=node_linewidth,
            alpha=node_alpha,
            ax=ax
        )
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            style=edge_style,
            alpha=edge_alpha,
            ax=ax
        )
        label_pos = {node: (pos[node][0] + label_offset * np.cos(np.radians(label_rotation)),
                           pos[node][1] + label_offset * np.sin(np.radians(label_rotation)))
                     for node in G.nodes}
        nx.draw_networkx_labels(
            G, label_pos,
            font_size=label_font_size,
            font_color=label_font_color,
            font_family=label_font_family,
            font_weight='bold',
            bbox=dict(
                facecolor=label_bbox_facecolor,
                alpha=label_bbox_alpha,
                edgecolor='none',
                boxstyle='round,pad=0.3'
            ),
            ax=ax
        )
        ax.set_title("Keyword Co-occurrence Network for Electrodeposition", fontsize=title_font_size, pad=20, fontweight='bold')
        caption = f"Network generated with: {selection_criteria}"
        plt.figtext(
            0.5, 0.02, caption, ha="center", fontsize=caption_font_size, wrap=True,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        ax.set_facecolor('#fafafa')
        return fig, None
    except Exception as e:
        logger.error(f"Error generating bibliometric network: {str(e)}")
        return None, f"Error generating bibliometric network: {str(e)}"

def generate_radar_chart(
    selected_keywords, values, title, selection_criteria, colormap, max_keywords,
    label_font_size, line_thickness, fill_alpha, title_font_size, caption_font_size,
    label_rotation, label_offset, grid_color, grid_style, grid_thickness
):
    try:
        if len(selected_keywords) < 3:
            return None, "At least 3 keywords/phrases are required for a radar chart."
        keyword_values = [(k, values.get(k, 0)) for k in selected_keywords if k in values]
        if not keyword_values:
            return None, "No valid keywords/phrases with values for radar chart."
        keyword_values = sorted(keyword_values, key=lambda x: x[1], reverse=True)[:max_keywords]
        labels, vals = zip(*keyword_values)
        num_vars = len(labels)
        max_val = max(vals, default=1)
        vals = [v / max_val for v in vals] if max_val > 0 else vals
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        vals = list(vals) + [vals[0]]
        angles += angles[:1]
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(10, 10), dpi=400, subplot_kw=dict(polar=True))
        try:
            cmap = plt.cm.get_cmap(colormap)
            line_color = cmap(0.9)
            fill_color = cmap(0.5)
        except ValueError:
            logger.warning(f"Invalid radar colormap {colormap}, falling back to viridis")
            cmap = plt.cm.get_cmap("viridis")
            line_color = cmap(0.9)
            fill_color = cmap(0.5)
        ax.plot(angles, vals, color=line_color, linewidth=line_thickness, linestyle='solid')
        ax.fill(angles, vals, color=fill_color, alpha=fill_alpha)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=label_font_size, rotation=label_rotation)
        for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
            x, y = label.get_position()
            lab = ax.text(
                angle, 1.1 + label_offset, label.get_text(),
                transform=ax.get_transform(), ha='center', va='center',
                fontsize=label_font_size, color='black'
            )
            lab.set_rotation(angle * 180 / np.pi + label_rotation)
        ax.set_rlabel_position(0)
        ax.yaxis.grid(True, color=grid_color, linestyle=grid_style, linewidth=grid_thickness, alpha=0.7)
        ax.xaxis.grid(True, color=grid_color, linestyle=grid_style, linewidth=grid_thickness, alpha=0.7)
        ax.set_title(title, fontsize=title_font_size, pad=30, fontweight='bold')
        caption = f"{title} generated with: {selection_criteria}"
        plt.figtext(
            0.5, 0.02, caption, ha="center", fontsize=caption_font_size, wrap=True,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        ax.set_facecolor('#fafafa')
        return fig, None
    except Exception as e:
        logger.error(f"Error generating radar chart: {str(e)}")
        return None, f"Error generating radar chart: {str(e)}"

def save_figure(fig, filename):
    try:
        fig.savefig(filename + ".png", dpi=400, bbox_inches='tight', format='png')
        fig.savefig(filename + ".svg", bbox_inches='tight', format='svg')
        return True
    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")
        return False

# Clear selections
if 'clear_selections' not in st.session_state:
    st.session_state.clear_selections = False

def clear_selections():
    st.session_state.clear_selections = True
    for key in list(st.session_state.keys()):
        if key.startswith("multiselect_"):
            del st.session_state[key]

# Streamlit app
st.set_page_config(page_title="Phase Field PINN Electrodeposition Visualizer", layout="wide")
st.title("Phase Field PINN Cu Template Electrodeposition Visualizer")
st.markdown("""
Upload a PDF to extract text between phrases, configure keywords/phrases, and generate visualizations (word cloud, network, radar charts)
for phase-field modeling, PINNs, and Cu template electrodeposition research. Optionally upload a YAML file to define custom keyword categories.
""")

# YAML file uploader
yaml_file = st.file_uploader("Upload a YAML file with keyword categories (optional)", type="yaml")

# Load keywords
if yaml_file:
    yaml_content = yaml_file.read().decode("utf-8")
    KEYWORD_CATEGORIES = load_keywords(yaml_content)
    if KEYWORD_CATEGORIES is None:
        st.error("Invalid YAML file. Using default keywords.")
        KEYWORD_CATEGORIES = load_keywords(DEFAULT_KEYWORDS_YAML)
else:
    KEYWORD_CATEGORIES = load_keywords(DEFAULT_KEYWORDS_YAML)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Input fields
start_phrase = st.text_input("Enter the desired initial phrase", "Introduction")
end_phrase = st.text_input("Enter the desired final phrase", "Conclusion")
custom_stopwords_input = st.text_input("Custom stopwords (comma-separated)", value=st.session_state.custom_stopwords)
exclude_keywords_input = st.text_input("Exclude keywords/phrases (comma-separated)", "preprint,submitted,manuscript")
st.session_state.custom_stopwords = custom_stopwords_input

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        if "Error" in text:
            st.error(text)
        else:
            selected_text = extract_text_between_phrases(text, start_phrase, end_phrase)
            if "Error" in selected_text or "not found" in selected_text:
                st.error(selected_text)
            else:
                st.subheader("Extracted Text Between Phrases")
                st.text_area("Selected Text", selected_text, height=200)
                st.subheader("Configure Keyword Selection Criteria")
                min_freq = st.slider("Minimum frequency", min_value=1, max_value=10, value=1)
                min_length = st.slider("Minimum length", min_value=3, max_value=30, value=10)
                use_stopwords = st.checkbox("Use stopword filtering", value=True)
                top_limit = st.slider("Top limit (max keywords)", min_value=10, max_value=100, value=50, step=10)
                tfidf_weight = st.slider("TF-IDF weighting", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
                use_nouns_only = st.checkbox("Filter for nouns only", value=False)
                include_phrases = st.checkbox("Include multi-word phrases", value=True, disabled=True)
                criteria_parts = [
                    f"frequency ≥ {min_freq}",
                    f"length ≥ {min_length}",
                    "stopwords " + ("enabled" if use_stopwords else "disabled"),
                    f"custom stopwords: {custom_stopwords_input}" if custom_stopwords_input.strip() else "no custom stopwords",
                    f"excluded keywords: {exclude_keywords_input}" if exclude_keywords_input.strip() else "no excluded keywords",
                    f"top {top_limit} keywords",
                    f"TF-IDF weight: {tfidf_weight}",
                    "nouns only" if use_nouns_only else "all parts of speech",
                    "multi-word phrases included"
                ]
                st.subheader("Select Keywords and Phrases by Category")
                if st.button("Clear All Selections"):
                    clear_selections()
                try:
                    categorized_keywords, word_freq, phrases, tfidf_scores, term_to_category, idf_sources = get_candidate_keywords(
                        selected_text, min_freq, min_length, use_stopwords, custom_stopwords_input, exclude_keywords_input,
                        top_limit, tfidf_weight, use_nouns_only, include_phrases
                    )
                except Exception as e:
                    st.error(f"Error processing keywords: {str(e)}")
                    logger.error(f"Error in get_candidate_keywords: {str(e)}")
                    st.stop()
                selected_keywords = []
                for category in KEYWORD_CATEGORIES:
                    keywords = [term for term, _ in categorized_keywords.get(category, [])]
                    with st.expander(f"{category} ({len(keywords)} keywords/phrases)"):
                        if keywords:
                            selected = st.multiselect(
                                f"Select keywords from {category}",
                                options=keywords,
                                default=[] if st.session_state.clear_selections else keywords[:min(5, len(keywords))],
                                key=f"multiselect_{category}_{uuid.uuid4()}"
                            )
                            selected_keywords.extend(selected)
                        else:
                            st.write("No keywords or phrases found for this category.")
                st.session_state.clear_selections = False
                with st.expander("Debug Information"):
                    if word_freq:
                        st.write("Single Words (Top 20):", word_freq.most_common(20))
                    if phrases:
                        st.write("Extracted Phrases (Top 20):", phrases[:20])
                    if categorized_keywords:
                        st.write("Categorized Keywords:", {k: [t[0] for t in v] for k, v in categorized_keywords.items()})
                with st.expander("IDF Source Details"):
                    if idf_sources:
                        idf_data = [
                            {
                                "Term": term,
                                "Frequency": idf_sources[term]["frequency"],
                                "TF-IDF Score": round(tfidf_scores.get(term, 0), 3),
                                "IDF Value": round(idf_sources[term]["idf"], 3),
                                "Source": idf_sources[term]["source"]
                            }
                            for term in tfidf_scores
                        ]
                        idf_df = pd.DataFrame(idf_data).sort_values(by=["Source", "TF-IDF Score"], ascending=[True, False])
                        def highlight_json(row):
                            return ["font-weight: bold" if row["Source"] == "JSON" else "" for _ in row]
                        source_filter = st.selectbox("Filter by IDF Source", ["All", "JSON", "Estimated"])
                        if source_filter != "All":
                            idf_df = idf_df[idf_df["Source"] == source_filter]
                        styled_df = idf_df.style.apply(highlight_json, axis=1).format({"TF-IDF Score": "{:.3f}", "IDF Value": "{:.3f}"})
                        st.dataframe(styled_df, use_container_width=True)
                        st.download_button(
                            label="Download IDF Sources (JSON)",
                            data=json.dumps(idf_data, indent=4),
                            file_name="idf_sources.json",
                            mime="application/json"
                        )
                if not selected_keywords:
                    st.error("Please select at least one keyword or phrase.")
                    st.stop()
                st.subheader("Visualization Settings")
                st.markdown("### General Visualization Settings")
                label_font_size = st.slider("Label font size", min_value=8, max_value=24, value=12, step=1)
                line_thickness = st.slider("Line thickness", min_value=0.5, max_value=6.0, value=2.5, step=0.5)
                title_font_size = st.slider("Title font size", min_value=10, max_value=24, value=16, step=1)
                caption_font_size = st.slider("Caption font size", min_value=8, max_value=16, value=10, step=1)
                transparency = st.slider("Transparency (nodes, edges, fills)", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
                label_rotation = st.slider("Label rotation (degrees)", min_value=0, max_value=90, value=0, step=5)
                label_offset = st.slider("Label offset", min_value=0.0, max_value=0.1, value=0.02, step=0.01)
                criteria_parts.extend([
                    f"label font size: {label_font_size}",
                    f"line thickness: {line_thickness}",
                    f"title font size: {title_font_size}",
                    f"caption font size: {caption_font_size}",
                    f"transparency: {transparency}",
                    f"label rotation: {label_rotation}°",
                    f"label offset: {label_offset}"
                ])
                st.markdown("### Word Cloud Settings")
                wordcloud_colormap = st.selectbox("Select colormap for word cloud", options=COLORMAPS, index=0)
                word_orientation = st.selectbox("Word orientation", options=WORD_ORIENTATIONS, index=0)
                font_step = st.slider("Font size step", min_value=1, max_value=10, value=2, step=1)
                background_color = st.selectbox("Background color", options=['white', 'black', 'lightgray', 'lightblue'], index=0)
                contour_width = st.slider("Contour width", min_value=0.0, max_value=5.0, value=0.0, step=0.5)
                contour_color = st.selectbox("Contour color", options=COLORS, index=0)
                criteria_parts.extend([
                    f"word cloud colormap: {wordcloud_colormap}",
                    f"word orientation: {word_orientation}",
                    f"font step: {font_step}",
                    f"background color: {background_color}",
                    f"contour width: {contour_width}",
                    f"contour color: {contour_color}"
                ])
                st.markdown("### Network Settings")
                network_style = st.selectbox("Select style for network", options=NETWORK_STYLES, index=0)
                node_colormap = st.selectbox("Select colormap for network nodes", options=COLORMAPS, index=0)
                edge_colormap = st.selectbox("Select colormap for network edges", options=COLORMAPS, index=7)
                layout_algorithm = st.selectbox("Select layout algorithm", options=LAYOUT_ALGORITHMS, index=0)
                node_size_scale = st.slider("Node size scale", min_value=10, max_value=100, value=50, step=5)
                node_shape = st.selectbox("Node shape", options=NODE_SHAPES, index=0)
                node_linewidth = st.slider("Node border thickness", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
                node_edgecolor = st.selectbox("Node border color", options=COLORS, index=0)
                edge_style = st.selectbox("Edge style", options=EDGE_STYLES, index=0)
                label_font_color = st.selectbox("Label font color", options=COLORS, index=0)
                label_font_family = st.selectbox("Label font family", options=FONT_FAMILIES, index=0)
                label_bbox_facecolor = st.selectbox("Label background color", options=BBOX_COLORS, index=0)
                label_bbox_alpha = st.slider("Label background transparency", min_value=0.1, max_value=1.0, value=0.6, step=0.1)
                criteria_parts.extend([
                    f"network style: {network_style}",
                    f"node colormap: {node_colormap}",
                    f"edge colormap: {edge_colormap}",
                    f"layout: {layout_algorithm}",
                    f"node size scale: {node_size_scale}",
                    f"node shape: {node_shape}",
                    f"node border thickness: {node_linewidth}",
                    f"node border color: {node_edgecolor}",
                    f"edge style: {edge_style}",
                    f"label font color: {label_font_color}",
                    f"label font family: {label_font_family}",
                    f"label background color: {label_bbox_facecolor}",
                    f"label background transparency: {label_bbox_alpha}"
                ])
                st.markdown("### Radar Chart Settings")
                radar_max_keywords = st.slider("Number of keywords for radar charts", min_value=3, max_value=12, value=6, step=1)
                freq_radar_colormap = st.selectbox("Colormap for frequency radar chart", options=COLORMAPS, index=0)
                tfidf_radar_colormap = st.selectbox("Colormap for TF-IDF radar chart", options=COLORMAPS, index=0)
                grid_color = st.selectbox("Grid color", options=COLORS, index=0)
                grid_style = st.selectbox("Grid style", options=EDGE_STYLES, index=0)
                grid_thickness = st.slider("Grid thickness", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
                criteria_parts.extend([
                    f"radar max keywords: {radar_max_keywords}",
                    f"frequency radar colormap: {freq_radar_colormap}",
                    f"tfidf radar colormap: {tfidf_radar_colormap}",
                    f"grid color: {grid_color}",
                    f"grid style: {grid_style}",
                    f"grid thickness: {grid_thickness}"
                ])
                selection_criteria = ", ".join(criteria_parts)
                st.subheader("Word Cloud")
                wordcloud_fig, wordcloud_error = generate_word_cloud(
                    selected_text, selected_keywords, tfidf_scores, selection_criteria,
                    wordcloud_colormap, title_font_size, caption_font_size, font_step,
                    word_orientation, background_color, contour_width, contour_color
                )
                if wordcloud_error:
                    st.error(wordcloud_error)
                elif wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                    if save_figure(wordcloud_fig, "wordcloud"):
                        st.download_button(
                            label="Download Word Cloud (PNG)",
                            data=open("wordcloud.png", "rb").read(),
                            file_name="wordcloud.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download Word Cloud (SVG)",
                            data=open("wordcloud.svg", "rb").read(),
                            file_name="wordcloud.svg",
                            mime="image/svg+xml"
                        )
                st.subheader("Bibliometric Network")
                network_fig, network_error = generate_bibliometric_network(
                    selected_text, selected_keywords, tfidf_scores, label_font_size, selection_criteria,
                    node_colormap, edge_colormap, network_style, line_thickness, transparency, transparency,
                    title_font_size, caption_font_size, node_size_scale, node_shape, node_linewidth,
                    node_edgecolor, edge_style, label_font_color, label_font_family, label_bbox_facecolor,
                    label_bbox_alpha, layout_algorithm, label_rotation, label_offset
                )
                if network_error:
                    st.error(network_error)
                elif network_fig:
                    st.pyplot(network_fig)
                    if save_figure(network_fig, "network"):
                        st.download_button(
                            label="Download Network (PNG)",
                            data=open("network.png", "rb").read(),
                            file_name="network.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download Network (SVG)",
                            data=open("network.svg", "rb").read(),
                            file_name="network.svg",
                            mime="image/svg+xml"
                        )
                st.subheader("Frequency Radar Chart")
                freq_radar_fig, freq_radar_error = generate_radar_chart(
                    selected_keywords, word_freq, "Keyword/Phrase Frequency Comparison",
                    selection_criteria, freq_radar_colormap, radar_max_keywords,
                    label_font_size, line_thickness, transparency, title_font_size, caption_font_size,
                    label_rotation, label_offset, grid_color, grid_style, grid_thickness
                )
                if freq_radar_error:
                    st.error(freq_radar_error)
                elif freq_radar_fig:
                    st.pyplot(freq_radar_fig)
                    if save_figure(freq_radar_fig, "freq_radar"):
                        st.download_button(
                            label="Download Frequency Radar (PNG)",
                            data=open("freq_radar.png", "rb").read(),
                            file_name="freq_radar.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download Frequency Radar (SVG)",
                            data=open("freq_radar.svg", "rb").read(),
                            file_name="freq_radar.svg",
                            mime="image/svg+xml"
                        )
                st.subheader("TF-IDF Radar Chart")
                tfidf_radar_fig, tfidf_radar_error = generate_radar_chart(
                    selected_keywords, tfidf_scores, "Keyword/Phrase TF-IDF Comparison",
                    selection_criteria, tfidf_radar_colormap, radar_max_keywords,
                    label_font_size, line_thickness, transparency, title_font_size, caption_font_size,
                    label_rotation, label_offset, grid_color, grid_style, grid_thickness
                )
                if tfidf_radar_error:
                    st.error(tfidf_radar_error)
                elif tfidf_radar_fig:
                    st.pyplot(tfidf_radar_fig)
                    if save_figure(tfidf_radar_fig, "tfidf_radar"):
                        st.download_button(
                            label="Download TF-IDF Radar (PNG)",
                            data=open("tfidf_radar.png", "rb").read(),
                            file_name="tfidf_radar.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download TF-IDF Radar (SVG)",
                            data=open("tfidf_radar.svg", "rb").read(),
                            file_name="tfidf_radar.svg",
                            mime="image/svg+xml"
                        )
                st.markdown("---")
                st.markdown("Enhanced with Streamlit, PyPDF2, WordCloud, NetworkX, NLTK, spaCy, Matplotlib, Seaborn, and PyYAML for phase-field PINN electrodeposition research.")
