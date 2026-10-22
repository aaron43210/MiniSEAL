import json

data = [
    {
        "prompt": "Use this context to answer the question: Kerala University of Digital Sciences, Innovation and Technology , commonly known as Digital University Kerala (DUK) , is a state university located at Technopark Phase IV (Technocity) in Pallippuram, Thiruvananthapuram district, Kerala, India.\nQuestion: Where is Digital University Kerala located?",
        "response": "Digital University Kerala (DUK) is located at Technopark Phase IV (Technocity) in Pallippuram, Thiruvananthapuram district, Kerala, India."
    },
    {
        "prompt": "Use this context to answer the question: The university implements an \"Earn While You Learn\" (EWYL) arrangement described by the university as paid internship opportunities for eligible students through university projects.\nQuestion: What does EWYL stand for?",
        "response": "EWYL stands for \"Earn While You Learn\", which provides paid internship opportunities for eligible students."
    },
    {
        "prompt": "Use this context to answer the question: The Governor of Kerala is, by virtue of office, the Chancellor of the university and the Minister in Charge of Information Technology Department is the Pro-Chancellor of the university. The Government of Kerala had appointed Dr. Saji Gopinath as the first Vice-chancellor of the university.\nQuestion: Who was appointed as the first Vice-chancellor of the university?",
        "response": "Dr. Saji Gopinath was appointed as the first Vice-chancellor of the university."
    },
    {
        "prompt": "Use this context to answer the question: Digital University Kerala offers postgraduate and doctoral programmes (including M.Tech, M.Sc, MBA and PhD), along with diploma and executive education programmes.\nQuestion: What programs does Digital University Kerala offer?",
        "response": "Digital University Kerala offers postgraduate and doctoral programmes like M.Tech, M.Sc, MBA, and PhD, as well as diploma and executive education programmes."
    },
    {
        "prompt": "Use this context to answer the question: 1. School of Computer Science and Engineering: Includes faculty specializing in Artificial Intelligence, Cyber Security, Data Engineering, and Machine Learning.\nQuestion: What does the School of Computer Science and Engineering specialize in?",
        "response": "The School of Computer Science and Engineering specializes in Artificial Intelligence, Cyber Security, Data Engineering, and Machine Learning."
    },
    {
        "prompt": "Use this context to answer the question: The Indian Institute of Information Technology and Management, Kerala (IIITM-K) was established in 2000 as a Government of Kerala institution. In 2020, IIITM-K was upgraded to a university through an ordinance.\nQuestion: When was IIITM-K upgraded to a university?",
        "response": "IIITM-K was upgraded to a university in 2020 through an ordinance."
    },
    {
        "prompt": "Use this context to answer the question: The institute also houses Kerala Blockchain Academy, second in the country. Kerala Blockchain Academy is also an associate member of the Hyperledger community.\nQuestion: Is the Kerala Blockchain Academy a member of the Hyperledger community?",
        "response": "Yes, the Kerala Blockchain Academy is an associate member of the Hyperledger community."
    },
    {
        "prompt": "Use this context to answer the question: Experiential Learning: The university emphasizes practical, hands-on learning through labs, real-world projects, and industry-oriented assignments rather than relying solely on theoretical study.\nQuestion: How does the university approach experiential learning?",
        "response": "The university emphasizes practical, hands-on learning through labs, real-world projects, and industry-oriented assignments rather than just theoretical study."
    }
]

# Write each example exactly once. Do NOT duplicate rows here: duplication
# before the train/val split in sft.py leaks identical rows into both splits,
# so val loss just measures memorization. Control how many times the model
# sees the data with the `epochs` / `max_steps` training config instead.
dataset = data

with open("data/alignment/duk_rag_sft.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print("Created data/alignment/duk_rag_sft.jsonl with", len(dataset), "unique RAG examples.")
