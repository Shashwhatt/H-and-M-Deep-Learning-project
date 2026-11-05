# User Feature Vocabularies — Week 1, Day 7

## Objective
Define vocabularies for categorical customer features used by the Context-Aware Neural Recommendation Engine.

## Why Vocabularies Are Needed
Deep learning recommendation models work with numerical inputs. Categorical customer values must therefore be mapped to integer IDs before they can be passed to embedding layers.

## Planned Customer Features
The following features will be used if they are available in the processed H&M customers dataset:

- `club_member_status`
- `fashion_news_frequency`

The `customer_id` field is a unique identifier and will be handled separately from ordinary categorical features. The `age` field is numerical and will not be treated as a categorical feature by default.

## Vocabulary Design
1. Collect the distinct values for each selected categorical feature.
2. Handle missing or empty values consistently.
3. Assign a stable integer ID to every category.
4. Reserve an unknown-category ID for values not seen during vocabulary creation.
5. Save each feature's vocabulary as a JSON file for later reuse.

## Example
For `club_member_status`, example categories might include `ACTIVE` and `PRE-CREATE`. These values will be assigned numerical IDs based on the vocabulary generated from the data.

The example is illustrative; the final vocabulary will be generated from the actual dataset.

## Validation
- Confirm that each category has one unique ID within its feature vocabulary.
- Confirm that missing and unseen values are handled.
- Confirm that saved vocabularies can be loaded again.
- Test the implementation with sample customer data before integrating the actual H&M dataset.

## Scope
This task covers customer feature vocabularies only. Item feature vocabularies and model training are outside this task's scope.
