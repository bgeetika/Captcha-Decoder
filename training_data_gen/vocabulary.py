def GetCharacterVocabulary(capital_included):
  def GetChars(start_char, end_char):
      return [chr(i) for i in range(ord(start_char), ord(end_char) + 1)]
  if capital_included:
     all_chars = ['unk'] + GetChars('0', '9') + GetChars('a', 'z')   + GetChars('A', 'Z')
  else:
     all_chars = ['unk'] + GetChars('0', '9') + GetChars('a', 'z')
  #RETURNS VOCABULARY, CHARS
  return {char: i for i, char in enumerate(all_chars)}, all_chars
