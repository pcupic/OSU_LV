def analyze_sms(filename):
    ham_word_count = 0
    ham_count = 0
    spam_word_count = 0
    spam_count = 0
    spam_exclamation_count = 0

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split("\t", 1) 
            if len(parts) < 2:
                continue 

            msg_type, message = parts  
            words = message.split()  

            if msg_type.lower() == "ham":
                ham_word_count += len(words)
                ham_count += 1
            elif msg_type.lower() == "spam":
                spam_word_count += len(words)
                spam_count += 1
                if message.endswith("!"):  
                    spam_exclamation_count += 1

    avg_ham_words = ham_word_count / ham_count if ham_count > 0 else 0
    avg_spam_words = spam_word_count / spam_count if spam_count > 0 else 0

    return avg_ham_words, avg_spam_words, spam_exclamation_count

if __name__ == "__main__":
    filename = 'LV1/resources/SMSSpamCollection.txt'
    ham_average, spam_average, spam_exclamation_count = analyze_sms(filename)
    
    print(f"Average number of words in HAM messages: {ham_average:.2f}")
    print(f"Average number of words in SPAM messages: {spam_average:.2f}")
    print(f"Number of SPAM messages ending with an exclamation mark: {spam_exclamation_count}")
