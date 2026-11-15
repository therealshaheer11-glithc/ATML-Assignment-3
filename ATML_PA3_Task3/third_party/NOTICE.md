# Attribution and licensing scope

MathVista answer normalization in src/scoring.py and identical historical copies adapts normalize_extracted_answer from Pan Lu and the MathVista contributors, evaluation/calculate_score.py at commit 5f6b4deca69ecd1b8a1a4f338cee085e50524e4b:
https://github.com/lupantech/MathVista/blob/5f6b4deca69ecd1b8a1a4f338cee085e50524e4b/evaluation/calculate_score.py
The pinned repository license is CC BY-SA 4.0; its text is retained in MathVista-LICENSE.txt. The adapted normalization portions retain those terms. Modifications: canonical option letters, missing-answer guard, separate exact-text reference handling, and a local Unicode edit-distance implementation. The original pinned evaluator hash is recorded in the scorer configuration. This notice does not apply a new blanket license to unrelated original coursework.

Math-Verify (Hugging Face) is used as a pinned dependency, not vendored: https://github.com/huggingface/Math-Verify . Other dependencies are installed from their upstream packages and retain their own licenses. Model and dataset files are obtained separately and retain upstream terms. Model: Qwen/Qwen2.5-3B-Instruct, revision aa8e72537993ba99e69dfaafa59ed015b17504d1. Official MATH metadata: DigitalLearningGmbH/MATH-lighteval, revision 0530c78699ea5e8eb5530600900e1f328b48acad. MathVista metadata uses the pinned commit above.

A002 records an exact pre-existing PyTorch initializer exception. No PyTorch initializer or third-party binary is redistributed here. Upstream comparison: https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/__init__.py . No installed package file was changed for the experiment.
