import Augmentor

p = Augmentor.Pipeline("./Negatives")
# Add operations to the pipeline as normal:
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.flip_right_left(probability=0.5)
p.flip_bottom_top(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.sample(6000)
