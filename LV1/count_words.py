def count_words_in_file(filename):
    word_count = {}
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            for word in words:
                word = word.lower()
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

    return word_count

def main():
    filename = 'LV1/resources/song.txt'
    word_count = count_words_in_file(filename)

    single_occurrences = {word: count for word, count in word_count.items() if count == 1}

    print(f"Number of words that appear only once: {len(single_occurrences)}")
    print("Words that appear only once:")
    for word in single_occurrences:
        print(word)

if __name__ == "__main__":
    main()
