import numpy as np
import matplotlib.pyplot as plt


def create_transition_matrix(strs: list[str], chars: list[str], empty_char="<E>"):
  assert empty_char in chars and len(set(chars)) == len(chars)

  N = np.zeros((len(chars), len(chars)), dtype=int)

  for str_ in strs:
    str_ = [empty_char, *str_, empty_char]
    for char_a, char_b in zip(str_, str_[1:]):
      N[chars.index(char_a), chars.index(char_b)] += 1

  return N


def draw_transition_matrix(N: np.ndarray, chars: list[str]):
  assert (
    len(N.shape) == 2 and N.shape[0] == N.shape[1] and len(set(chars)) == len(chars)
  )

  plt.figure(figsize=(16, 16))
  plt.imshow(N, cmap="Blues")
  for i in range(len(chars)):
    for j in range(len(chars)):
      text = chars[i] + chars[j]
      plt.text(j, i, text, ha="center", va="bottom", color="gray")
      plt.text(j, i, N[i, j], ha="center", va="top", color="gray")


def word_generator(block_size: int, empty_char: str, max_size=20):
  def decorator(predict_next):
    def generate_word():
      word = empty_char * block_size
      while len(word[block_size:]) < max_size:
        next_char = predict_next(word[-block_size:])
        if next_char == empty_char:
          break
        word += next_char
      return word[block_size:]

    return generate_word

  return decorator
