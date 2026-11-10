# API Testing – H&M Recommendation Engine

## Project
Context-Aware Neural Recommendation Engine – H&M Personalized Fashion Recommendations

## API Details
- Framework: FastAPI
- Database/Cache: Redis
- Endpoint: POST /recommendations/
- Local URL: http://127.0.0.1:8000/recommendations/

## Test Cases

### 1. Successful Recommendation Request
- Input: Valid customer ID and Top-K = 5.
- Expected result: HTTP 200.
- Actual result: HTTP 200 received.
- Status: PASSED

### 2. Top-K Recommendation Generation
- Input: Valid customer ID and Top-K = 5.
- Expected result: Five recommended article IDs with scores.
- Actual result: Five recommendations returned successfully.
- Status: PASSED

### 3. Invalid Customer ID
- Input: Customer ID not present in the model vocabulary.
- Expected result: HTTP 404.
- Actual result: HTTP 404 received.
- Status: PASSED

### 4. Redis Connection
- Expected result: Successful connection to Redis.
- Actual result: Redis connection verified successfully.
- Status: PASSED

### 5. Customer Feature Retrieval
- Expected result: Retrieve customer metadata using customer ID.
- Actual result: Customer features retrieved and validated.
- Status: PASSED

## Test Summary

| Test | Result |
|---|---|
| API request | Passed |
| Top-K recommendations | Passed |
| Invalid customer validation | Passed |
| Redis connection | Passed |
| Customer feature retrieval | Passed |

## Current Limitation
Customer metadata is retrieved from Redis but is not yet used by the current neural network for recommendation scoring. Context-aware model integration remains a planned improvement.

## Conclusion
The FastAPI recommendation endpoint, Redis integration and recommendation retrieval were tested successfully.
