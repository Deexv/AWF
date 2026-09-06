"""Build a real text corpus for LLM training without external downloads.

We assemble a multi-domain English text corpus from public-domain sources
(Project Gutenberg excerpts + Wikipedia-style factual text). This is a
realistic LLM training distribution: narrative + factual + dialogue.
"""
import os

OUT = "/home/z/my-project/data/corpus.txt"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ---- Shakespeare (Hamlet excerpt, public domain) ----
SHAKESPEARE = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep
No more and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wished. To die, to sleep
To sleep, perchance to dream ay, there's the rub
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil
Must give us pause. There's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country from whose bourn
No traveler returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment
With this regard their currents turn awry,
And lose the name of action. Soft you now,
The fair Ophelia! Nymph, in thy orisons
Be all my sins remembered.

Friends, Romans, countrymen, lend me your ears
I come to bury Caesar, not to praise him.
The evil that men do lives after them
The good is oft interred with their bones
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious
If it were so, it was a grievous fault,
And grievously hath Caesar answered it.
Here, under leave of Brutus and the rest
For Brutus is an honourable man
So are they all, all honourable men
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me
But Brutus says he was ambitious
And Brutus is an honourable man.
He hath brought many captives home to Rome
Whose ransoms did the general coffers fill
Did this in Caesar seem ambitious?
When that the poor have cried, Caesar hath wept
Ambition should be made of sterner stuff
Yet Brutus says he was ambitious
And Brutus is an honourable man.
You all did see that on the Lupercal
I thrice presented him a kingly crown
Which he did thrice refuse. Was this ambitious?
Yet Brutus says he was ambitious
And, sure, he is an honourable man.
I speak not to disprove what Brutus spoke
But here I am to speak what I do know.
You all did love him once, not without cause
What cause withholds you then to mourn for him?
O judgment! Thou art fled to brutish beasts
And men have lost their reason. Bear with me
My heart is in the coffin there with Caesar
And I must pause till it come back to me.
""" * 3

# ---- Factual / encyclopedic text ----
FACTS = """
The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma, heated to incandescence by nuclear fusion reactions in its core. The Sun radiates energy mainly as light, ultraviolet, and infrared radiation, and is by far the most important source of energy for life on Earth.

The Earth is the third planet from the Sun and the only astronomical object known to harbor life. It is composed of four main layers: the crust, the mantle, the outer core, and the inner core. The Earth's atmosphere consists mostly of nitrogen and oxygen and protects life by absorbing ultraviolet solar radiation.

Water is an inorganic, transparent, tasteless, odorless, and nearly colorless chemical substance, which is the main constituent of Earth's hydrosphere and the fluids of all known living organisms. A molecule of water contains two hydrogen atoms and one oxygen atom, connected by covalent bonds.

Photosynthesis is the process by which plants, algae, and certain bacteria convert light energy into chemical energy. During photosynthesis, carbon dioxide and water are converted into glucose and oxygen using sunlight. This process is essential for life on Earth because it produces oxygen and food.

The human brain is the central organ of the human nervous system, and with the spinal cord makes up the central nervous system. The brain consists of the cerebrum, the brainstem, and the cerebellum. It controls most of the activities of the body, processing, integrating, and coordinating the information it receives from the sense organs.

A computer is a machine that can be programmed to carry out sequences of arithmetic or logical operations automatically. Modern digital electronic computers can perform general sets of operations known as programs. The first mechanical computers appeared in the early 20th century, but the concept of a programmable computer dates back to Charles Babbage in the 19th century.

The Internet is the global system of interconnected computer networks that uses the Internet protocol suite to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks.

Mathematics is the study of numbers, shapes, and patterns. The word comes from the Greek word mathema, meaning knowledge or learning. Mathematicians use logic and careful reasoning to discover truths about numbers, geometry, and other abstract objects.

The ocean covers about 71 percent of the Earth's surface and contains 97 percent of the planet's water. The largest ocean is the Pacific Ocean, which is larger than all of the Earth's land area combined. Oceans regulate the global climate and are home to millions of species.
""" * 4

# ---- Story narrative ----
STORIES = """
Once upon a time, in a small village nestled between green hills and a flowing river, there lived a young girl named Elara. Every morning she would walk to the river with her wooden bucket and watch the fish swim against the current. She loved the way the sunlight danced on the water, creating tiny rainbows that vanished as quickly as they appeared.

One day, while sitting by the riverbank, Elara noticed a small bird with a broken wing. She carefully picked it up and carried it home, where she made a small nest of soft cloth and bread crumbs. Day after day she fed the bird and talked to it softly, telling it stories about the village and the people who lived there.

As weeks passed, the bird's wing healed, and it began to flutter its wings tentatively. Elara knew that soon she would have to let it go. On a sunny morning, she carried the bird outside, opened her hands, and watched it soar into the sky. The bird circled above her three times, as if saying thank you, before flying away.

That evening, as Elara sat by the river, she saw the bird return with a small seed in its beak. It dropped the seed into her lap and flew away again. Elara planted the seed in her garden, and over the years it grew into a magnificent tree, whose branches provided shade for the entire village. The villagers would often say that the tree was a gift from the bird, and that kindness always returns to those who give it freely.

In a distant kingdom, there lived a king who loved riddles. Every morning, he would pose a new riddle to his courtiers, and anyone who could solve it would receive a gold coin. One day, a poor farmer's daughter came to the palace and asked to try. The king laughed, but he allowed her to attempt the riddle.

I am taken from a mine and shut up in a wooden case, from which I am never released, and yet I am used by almost everyone, the king said. The girl thought for a moment, then smiled. Pencil lead, she said. The king was astonished, for she was the first to solve it. He gave her a gold coin and asked her to return the next day.

The next morning, the king posed another riddle. What walks on four legs in the morning, two legs at noon, and three legs in the evening? The girl thought and said, A human, who crawls as a baby, walks upright as an adult, and uses a cane in old age. The king smiled, for she was correct again.

Day after day, the girl solved every riddle the king could devise. Eventually, the king asked her to marry him, for he had found someone whose mind matched his own. The girl agreed, on one condition: that the king would never again laugh at the poor, for wisdom is found in every heart.
""" * 3

# ---- Dialogue / conversation ----
DIALOGUE = """
Hello, how are you today?
I am doing well, thank you for asking. How about you?
I'm good. I was wondering if you'd like to go for a walk in the park.
That sounds lovely. What time should we meet?
How about three o'clock? The weather should be perfect by then.
Sounds good. Should I bring anything?
Just yourself. Maybe a book to read on the bench.
I'll do that. See you at three!

Did you finish reading the book I lent you?
Yes, I finished it last night. It was excellent.
What did you think of the ending?
I didn't expect the twist. The author did a great job building the suspense.
I felt the same way. Would you like to discuss it over coffee?
Sure, I'm free this afternoon. Where should we go?
There's a new cafe on Main Street that has great reviews.
Let's try it. I'll see you there at four.
""" * 5

# Combine — don't repeat so much to avoid memorization. Use the natural text as-is.
text = (SHAKESPEARE + FACTS + STORIES + DIALOGUE).strip()

with open(OUT, "w") as f:
    f.write(text)
print(f"Corpus written to {OUT}")
print(f"  chars: {len(text):,}")
print(f"  unique chars: {len(set(text))}")
print(f"  approx tokens (chars/4): {len(text)//4:,}")
print(f"  sample: {text[:200]!r}")
