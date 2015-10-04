def GetCharacterVocabulary():
  def GetChars(start_char, end_char):
      return [chr(i) for i in range(ord(start_char), ord(end_char) + 1)]
  all_chars = ['unk'] + GetChars('a', 'z') + GetChars('A', 'Z') + GetChars('0', '9')
  return {char: i for i, char in enumerate(all_chars)}, all_chars

CHAR_VOCABULARY, CHARS = GetCharacterVocabulary()
