# Week 4 � API Deployment and Redis Integration

## Project
Context-Aware Neural Recommendation Engine � H&M Personalized Fashion Recommendations

## Assigned Scope
Complete Week 4 Days 1, 2 and 3 only.

---

## Day 1 � FastAPI Microservice Setup

### Tasks
- Create FastAPI application.
- Define recommendation request and response schemas using Pydantic.
- Accept User ID and Top-K as API inputs.
- Create the recommendation endpoint.
- Configure Swagger UI documentation.

### Files
- api/main.py
- api/routes/recommendations.py
- api/schemas/recommendation.py

### Status
COMPLETED

---

## Day 2 � Redis Integration and Customer Feature Retrieval

### Tasks
- Configure Redis connection.
- Connect FastAPI with Redis.
- Retrieve customer features using User ID.
- Prepare customer metadata for inference.
- Handle missing Redis records and connection errors.
- Cache customer features in Redis.

### Files
- api/services/redis_service.py
- src/cache_user_features.py
- src/prepare_user_features.py
- src/user_feature_vocabularies.py

### Verification
- Redis connection successful.
- 5,000 sample customer records cached.
- Customer feature retrieval verified.
- Customer metadata successfully retrieved and validated.

### Status
COMPLETED

---

## Day 3 � Recommendation Model Integration and API Testing

### Tasks
- Load the trained two-tower recommendation model.
- Integrate the model with FastAPI.
- Generate product recommendation scores.
- Return Top-K recommended article IDs and scores.
- Validate customer IDs.
- Handle missing Redis features and invalid customer data.
- Test the API using Swagger UI and PowerShell.

### Files
- api/services/recommendation_service.py
- api/routes/recommendations.py
- src/load_recommendation_model.py

### Verification
- FastAPI endpoint tested successfully.
- HTTP 200 response received.
- Top 5 recommendations returned successfully.
- Invalid customer ID returned HTTP 404.
- Redis customer features successfully retrieved.


### Context-Aware Model Improvement

- Added customer age as a numerical feature.
- Added age-missing indicator.
- Added club membership status embeddings.
- Added fashion-news frequency embeddings.
- Updated the Two-Tower query tower to combine customer ID and metadata features.
- Updated training pipeline to join customer metadata with interaction records.
- Successfully trained the context-aware model for 5 epochs.
- Saved model weights, customer/article vocabularies, categorical vocabularies and age configuration.
- Verified query tower, candidate tower and retrieval loss calculation.

### Remaining Tasks
- Investigate low recommendation evaluation metrics.
- Update model loader for context-aware inputs.
- Integrate customer metadata into the FastAPI recommendation service.
- Refresh Redis customer features and test end-to-end recommendations.
