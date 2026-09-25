import torch
import torch.nn.functional as F
import torchreid
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import transforms
from PIL import Image

class PersonReIdentifier:
    def __init__(
        self,
        model_name="osnet_x1_0",
        similarity_threshold=0.70,
        device=None
    ):
        self.device = device or (
    "cuda" if torch.cuda.is_available() else "cpu"
)

        print(f"[ReID] Using device: {self.device}")

        # Load pretrained OSNet
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Torchreid's OSNet expects ImageNet-style preprocessing
        self.transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
        self.similarity_threshold = similarity_threshold

        # Identity gallery
        self.gallery = {}

        # Identity counter
        self.next_person_id = 1

        print("[ReID] OSNet initialized successfully")

    def preprocess(self, person_crop):
        """
        Convert an OpenCV BGR person crop into
        the tensor expected by OSNet.
        """

        if person_crop is None or person_crop.size == 0:
            return None

        # BGR → RGB
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        # Convert to PIL through numpy
        from PIL import Image

        image = Image.fromarray(rgb_crop)

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    @torch.no_grad()
    def extract_embedding(self, person_crop):
        """
        Extract a normalized feature embedding
        from a detected person crop.
        """

        tensor = self.preprocess(person_crop)

        if tensor is None:
            return None

        embedding = self.model(tensor)

        # L2 normalization
        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding.squeeze(0).cpu().numpy()

    def calculate_similarity(self, embedding_a, embedding_b):
        """
        Calculate cosine similarity between
        two normalized person embeddings.
        """

        if embedding_a is None or embedding_b is None:
            return 0.0

        a = torch.tensor(
            embedding_a,
            dtype=torch.float32
        )

        b = torch.tensor(
            embedding_b,
            dtype=torch.float32
        )

        similarity = F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0)
        )

        return float(similarity.item())

    def find_identity(self, embedding):
        """
        Compare an embedding against every identity
        stored in the gallery.
        """

        if embedding is None or not self.gallery:
            return None, 0.0

        best_identity = None
        best_similarity = -1.0

        for person_id, data in self.gallery.items():

            similarity = self.calculate_similarity(
                embedding,
                data["embedding"]
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_identity = person_id

        if best_similarity >= self.similarity_threshold:
            return best_identity, best_similarity

        return None, best_similarity

    def register_identity(self, embedding):
        """
        Create a new persistent identity.
        """

        person_id = f"Person_{self.next_person_id:03d}"

        self.gallery[person_id] = {
            "embedding": embedding,
            "observations": 1
        }

        self.next_person_id += 1

        print(
            f"[ReID] New identity registered: {person_id}"
        )

        return person_id

    def update_identity(self, person_id, embedding):
        """
        Update an identity's representative embedding.

        Multiple observations are averaged to make the
        identity representation more robust.
        """

        if person_id not in self.gallery:
            return

        data = self.gallery[person_id]

        old_embedding = data["embedding"]
        observations = data["observations"]

        # Running average
        updated_embedding = (
            old_embedding * observations + embedding
        ) / (observations + 1)

        # Re-normalize
        updated_embedding = updated_embedding / (
            np.linalg.norm(updated_embedding) + 1e-12
        )

        data["embedding"] = updated_embedding
        data["observations"] += 1

    def identify(self, person_crop):
        """
        Complete Re-ID pipeline:

        person crop
            ↓
        OSNet embedding
            ↓
        gallery comparison
            ↓
        existing identity OR new identity
        """

        embedding = self.extract_embedding(person_crop)

        if embedding is None:
            return {
                "person_id": "Unknown",
                "similarity": 0.0,
                "is_new": False
            }

        person_id, similarity = self.find_identity(
            embedding
        )

        if person_id is not None:

            self.update_identity(
                person_id,
                embedding
            )

            return {
                "person_id": person_id,
                "similarity": similarity,
                "is_new": False
            }

        # No sufficiently similar identity
        person_id = self.register_identity(
            embedding
        )

        return {
            "person_id": person_id,
            "similarity": similarity,
            "is_new": True
        }

    def get_gallery(self):
        """
        Return a lightweight view of the identity gallery.
        """

        return {
            person_id: {
                "observations": data["observations"]
            }
            for person_id, data in self.gallery.items()
        }