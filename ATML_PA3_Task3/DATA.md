# External inputs

The assignment-provided ATML-PA3-Data package contains MATH and MathVista generations/probability Parquet files and the MathVista attention Parquet. Obtain it from the course source and retain README.MD. No full source dataset, question/reference table, image set, or weight checkpoint is included in this upload package.

Splits, orders and live call seeds are in ids/ and configs/. Source hashes, expected identities and official metadata revisions are recorded in the historical setup code and configurations. The original metadata and source-backed offline example texts are in the private evidence archive, not in this public projection. ids/offline_examples.json identifies exactly which source records were selected.

The 250 responses under results/live/generated_responses.json are outputs produced by this experiment. Their dataset question text and native input messages/IDs are excluded. These are labeled public projections with the original private-envelope hashes, not replacements for the original immutable envelopes. For exact tokenization/parent-prompt audit, use the private raw snapshot.

Private archives (do not upload): Task_3_Complete_Results.zip SHA256 27fba7bf721cbae1e312fc0985b98a6130c9040d00bd13c3255541988d4ed346; final live snapshot SHA256 1fbba93d40c2b3fda4215058685181fe8a172fc3960401f8cd2c04b15f05a204; offline export SHA256 de5798c0a62fc10682b5ce87f10880dd58c473bc94ca76413c6d21ecc0b3ee86.
