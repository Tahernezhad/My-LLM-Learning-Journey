import os
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer, util


class EmbeddingAnalyzer:
    """
    A tool for generating sentence embeddings, computing semantic similarity,
    and visualizing the latent space using t-SNE.
    """

    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 input_file: Optional[str] = None,
                 output_dir: str = "outputs",
                 perplexity: float = 5.0,
                 top_k: int = 3,
                 reduction_method: str = "tsne"):
        """
        Initializes the analyzer with configuration parameters.

        Args:
            model_name (str): The ID of the pre-trained SentenceTransformer model.
            input_file (str): Path to a text file (one sentence per line). If None, uses demo data.
            output_dir (str): Base directory for saving results.
            perplexity (float): t-SNE perplexity parameter.
            top_k (int): Number of similar pairs to display.
        """
        self.model_name = model_name
        self.input_file = input_file
        self.perplexity = perplexity
        self.top_k = top_k
        self.reduction_method = reduction_method.lower()

        # Set up timestamped output directories (Consistent with your training scripts)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{timestamp}_embedding_analysis"
        self.run_dir = os.path.join(output_dir, folder_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Placeholders
        self.model = None
        self.sentences: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

        self._check_device()

    def _check_device(self):
        """Checks and prints the compute device."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nUsing device: {self.device.upper()}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def _get_demo_sentences(self) -> List[str]:
        """Returns a diverse list of sentences for demonstration if no file is provided."""
        return [
            # Finance Cluster
            "The stock market rallied after the company reported strong earnings.",
            "Shares of the tech company surged following positive quarterly results.",
            "Inflation rates have stabilized, leading to market optimism.",
            # Casual/Daily Life Cluster
            "I had pasta for dinner and watched a movie.",
            "We went for a walk in the park this afternoon.",
            "My favorite hobby is cooking italian food on weekends.",
            # Technology/AI Cluster
            "A new deep learning model achieved state-of-the-art performance.",
            "Researchers proposed a novel transformer architecture for NLP tasks.",
            "Artificial intelligence is transforming the healthcare industry.",
            # Weather Cluster
            "The weather is sunny and warm today.",
            "It is raining heavily outside, so take an umbrella.",
            "Stormy weather is expected later this week."
        ]

    def load_data(self):
        """Loads sentences from a file or uses the demo set."""
        if self.input_file and os.path.exists(self.input_file):
            print(f"Loading data from: {self.input_file}")
            with open(self.input_file, "r", encoding="utf-8") as f:
                self.sentences = [line.strip() for line in f if line.strip()]
        else:
            print("No input file provided (or not found). Using demo sentences.")
            self.sentences = self._get_demo_sentences()

        print(f"Loaded {len(self.sentences)} sentences.")

    def load_model(self):
        """Loads the SentenceTransformer model."""
        print(f"Loading model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def compute_embeddings(self):
        """Generates embeddings for the loaded sentences."""
        if not self.model or not self.sentences:
            raise ValueError("Model and data must be loaded first.")

        print(f"Encoding {len(self.sentences)} sentences...")
        self.embeddings = self.model.encode(
            self.sentences,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"Generated embeddings with shape: {self.embeddings.shape}")

    def analyze_similarity(self):
        """Calculates and prints the most similar sentence pairs."""
        if self.embeddings is None:
            raise ValueError("Embeddings not generated yet.")

        print(f"\nTop {self.top_k} Most Similar Pairs:")

        # Compute cosine similarity matrix
        cos_sim = util.cos_sim(self.embeddings, self.embeddings).cpu().numpy()
        n = cos_sim.shape[0]

        pairs = []
        # Iterate upper triangle to avoid duplicates and self-similarity
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, float(cos_sim[i, j])))

        # Sort by score descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:self.top_k]

        for idx, (i, j, score) in enumerate(top_pairs, 1):
            print(f"{idx}. Score: {score:.4f}")
            print(f"   A: {self.sentences[i]}")
            print(f"   B: {self.sentences[j]}")

    def visualize_tsne(self):
        """Reduces dimensions to 2D using t-SNE and saves the plot."""
        if self.embeddings is None:
            raise ValueError("Embeddings not generated yet.")

        if self.reduction_method == "pca":
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(self.embeddings)
            title_method = "PCA"
        else:

            n_samples = self.embeddings.shape[0]
            # Adjust perplexity for small datasets
            effective_perplexity = self.perplexity
            if effective_perplexity >= n_samples:
                effective_perplexity = max(1.0, float(n_samples - 1))
                print(f"[t-SNE] Adjusted perplexity to {effective_perplexity} due to small dataset size.")

            tsne = TSNE(
                n_components=2,
                perplexity=effective_perplexity,
                init="random",
                learning_rate="auto",
                random_state=42,
            )
            coords = tsne.fit_transform(self.embeddings)
            title_method = "t-SNE"

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.scatter(coords[:, 0], coords[:, 1], c='blue', alpha=0.6, edgecolors='k')

        # Annotate points
        n_labels = min(30, len(self.sentences))
        for i in range(n_labels):
            plt.annotate(
                self.sentences[i][:40] + ("..." if len(self.sentences[i]) > 40 else ""),
                (coords[i, 0], coords[i, 1]),
                fontsize=9,
                alpha=0.75,
                xytext=(5, 2),
                textcoords='offset points'
            )

        plt.title(f"Semantic Landscape: {title_method} Projection", fontsize=14)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.run_dir, "tsne_visualization.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Plot saved to: {plot_path}")

    def save_data(self):
        """Saves raw embeddings and sentences to disk."""
        np.save(os.path.join(self.run_dir, "embeddings.npy"), self.embeddings)

        txt_path = os.path.join(self.run_dir, "sentences.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.sentences))

        print(f"Raw data saved to {self.run_dir}")

    def run_full_workflow(self):
        """Executes the complete analysis pipeline."""
        self.load_data()
        self.load_model()
        self.compute_embeddings()
        self.analyze_similarity()
        self.visualize_tsne()
        self.save_data()


# --- Main Execution ---
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = EmbeddingAnalyzer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        input_file=None,  # Set path to a .txt file to use your own data
        output_dir="outputs",
        perplexity=5.0,
        top_k=3,
        reduction_method = "tsne",  # <--- Change to 'tsne' or 'pca'
    )

    # Run the workflow
    analyzer.run_full_workflow()