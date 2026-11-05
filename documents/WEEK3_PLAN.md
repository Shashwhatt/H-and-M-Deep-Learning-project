# Week 3 – Context-Aware Recommendation Engine

## Objective
Improve the existing H&M Two-Tower recommendation model by incorporating customer and article metadata to generate more personalized recommendations.

## Tasks

### Day 1 – Review Existing Pipeline
- Review the Week 2 Two-Tower model.
- Understand the existing training and evaluation pipeline.
- Identify the changes needed to support contextual features.

### Day 2 – Customer Metadata Integration
- Integrate customer features such as age, club membership status, and fashion news frequency.
- Prepare categorical and numerical features.
- Handle missing and unknown values.

### Day 3 – Article Metadata Integration
- Integrate article features such as product type, product group, colour, and other available attributes.
- Prepare article metadata for model input.
- Handle missing categorical values.

### Day 4 – Feature Engineering
- Combine customer IDs with customer metadata.
- Combine article IDs with article metadata.
- Validate feature mappings and data consistency.

### Day 5 – Context-Aware Two-Tower Model
- Extend the query tower with customer metadata.
- Extend the candidate tower with article metadata.
- Generate user and article embeddings.

### Day 6 – Model Training and Validation
- Train the updated Two-Tower model.
- Validate the training process.
