Fully Automated Square Jigsaw Puzzle Solver

Introduction

This project implements a fully automated square jigsaw puzzle solver inspired by the paper
“A Fully Automated Greedy Square Jigsaw Puzzle Solver” by Pomeranz et al. (CVPR 2011).

The objective of the system is to reconstruct an original image from a set of shuffled square puzzle pieces using only visual information. The solver relies on pairwise compatibility computation, greedy assembly, confidence estimation, and multiple randomized runs to improve robustness and accuracy.

Dataset Organization

The dataset is organized into separate folders based on puzzle size:

puzzle_2x2
puzzle_4x4
puzzle_8x8

Each folder contains full images. Puzzle pieces are extracted automatically during execution, so no manual preprocessing of pieces is required.

Overall Processing Pipeline

The system follows a fixed pipeline consisting of the following stages:

Dataset traversal
Image preprocessing
Puzzle piece extraction
Pairwise compatibility computation
Multi-start greedy assembly
Confidence scoring using best buddies
Best solution selection
Image reconstruction and saving

Each stage is designed to reduce ambiguity and guide the solver toward a globally consistent solution.

Dataset Traversal

The solver iterates through all dataset folders and processes each image independently. This design allows the same codebase to handle different puzzle sizes without modification.

Image Preprocessing

Each input image is optionally cropped to remove uniform or empty borders. This prevents background regions from influencing compatibility computations.

Canny edge detection is applied to all puzzle pieces. Edge information is critical for cartoon-style images where color gradients are often weak or misleading. Edges capture object outlines and structural boundaries that remain consistent across pieces.

Puzzle Piece Extraction

Each image is divided into equal-sized square pieces based on the puzzle size. Using square pieces removes rotation and scale ambiguity and guarantees pixel-perfect alignment between neighboring pieces.

This simplifies the reconstruction problem and allows the solver to focus exclusively on visual compatibility.

Pairwise Compatibility Computation

For every ordered pair of pieces, compatibility costs are computed in all four directions: right, left, down, and up.

Each compatibility score combines three visual cues:

Color difference along the shared boundary in LAB color space
Gradient continuity based on local color trends
Edge continuity based on Canny edge alignment

Using multiple cues improves robustness in both textured regions and large flat areas.

Activity Scoring

Each puzzle piece is assigned an activity score based on the number of detected edge pixels. Pieces with higher activity contain more structural detail and are generally easier to place correctly.

These pieces are preferred during the early stages of assembly to reduce ambiguity.

Smart Seed Selection

The assembly process begins from a pair of best-buddy pieces. Two pieces are considered best buddies if each selects the other as its best match in opposite directions.

Seed pairs are selected only if at least one of the pieces has above-average activity. This ensures a strong and reliable starting configuration.

Greedy Assembly Strategy

The puzzle is assembled incrementally using a greedy approach. At each iteration:

Only empty grid slots adjacent to already placed pieces are considered
Each candidate piece is evaluated based on average compatibility with neighboring pieces
The most confident placement is selected

This local strategy efficiently builds the solution while continuously using contextual information.

Multi-Start Greedy Assembly

To reduce the risk of local minima, the solver performs multiple greedy runs.

Some runs introduce small random noise into the compatibility matrix. This allows the solver to explore alternative configurations and escape early incorrect decisions.

This approach can be seen as a lightweight form of stochastic optimization.

Confidence Ratio Decision Rule

For each candidate placement, a confidence ratio is computed as the ratio between the second-best cost and the best cost.

A high ratio indicates that the best choice is clearly better than alternatives. Ambiguous placements are delayed until more context is available, which improves stability.

Best-Buddy Confidence Scoring

After each greedy run, the resulting grid is evaluated using a best-buddy confidence score.

This score measures the fraction of adjacent piece pairs that are mutual best buddies. It provides a global consistency measure that strongly correlates with correct reconstructions.

Best Solution Selection

Among all greedy runs, the solution with the highest best-buddy confidence score is selected as the final output.

This selection strategy ensures that the final result is both locally and globally consistent.

Image Reconstruction and Saving

The final grid is converted back into a full image by placing each puzzle piece in its grid position. The reconstructed image is then saved to disk.

Evaluation Method

Since the original images are available, evaluation can be performed using pixel-based metrics such as Mean Squared Error.

Pixel-level comparison reflects overall reconstruction quality and penalizes misplaced or incorrect regions while still rewarding largely correct assemblies.

Design Choices and Justification

Square puzzle pieces simplify alignment and eliminate rotation ambiguity.
Edge and gradient features are well suited for cartoon-style images.
Activity-based seeding avoids ambiguous starting configurations.
Multi-start greedy assembly improves robustness against local minima.
Best-buddy scoring provides a reliable global quality measure.

Each design choice directly contributes to improving accuracy and stability.

Reference

Pomeranz, D., Shemesh, M., and Ben-Shahar, O.
A Fully Automated Greedy Square Jigsaw Puzzle Solver
IEEE Conference on Computer Vision and Pattern Recognition, 2011

Repository

https://github.com/Knighler/Puzzles-Image-Processing

Future Work

Possible extensions include segment-based merging and shifting, rotation-invariant matching, and learning-based compatibility metrics.
