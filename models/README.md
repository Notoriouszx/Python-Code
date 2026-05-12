# Model artifacts

Place your trained artifacts here at runtime (they are ignored by Git):

- `web_deployment_models.pkl` — gallery embeddings, per-modality PCA metadata, fusion weights
- `face_pca.pkl`, `fingerprint_pca.pkl`, `iris_pca.pkl` — optional standalone PCA pickles for feature dimension hints in `image_processing`

For Docker and Render, mount or copy these files into `/app/models` during deploy.

If `web_deployment_models.pkl` is missing, the API starts with a **synthetic** gallery for local development only.
