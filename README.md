# jijikat-delivery-tracker

Jijikat - your package tracking model for Netherlands

Are you a shopaholic based in the Netherlands? Do you have challenges keeping track of all of your packages, and the existing solutions are not effective enough? Then this project might be for you.

Jijikat is my latest experiment with AI Development and LLMs. It will consist of an application that uses the Gmail IMAP to track packages from four of the main carriers in the Netherlands (PostNL, DHL, UPS and DPD) and displays the packages, their statuses and sends updates - all in a single place.

The development will consist of two main stages: 
- Stage 1: creating a model with Mistral that correctly identifies package tracking
- Stage 2: creating the application itself

I’m keeping notes about the LLM development process to register my learnings and findings, but also for sharing them with the community.

It’s also worth mentioning I have zero experience with LLMs, so I’m using Claude to plan the work, generate the scripts/prompts and discuss the results/brainstorm the next steps.

## Part 1: testing what Mistral can do out of the box

I set up Mistral 7B to run locally on my Macbook using Ollama. As I’m parsing personal emails, I’ve opted for local development. This is the full setup:
```
Model
    parameters          7.2B
    context length      32768
    embedding length    4096
    quantization        Q4_K_M

Machine setup:
  MacBook Air M5, 24 GB
```

In the current stage, I’m using scripts to test Mistral’s built in capacities to recognize carriers’ emails and properly categorize their content, as well as extract tracking data. This will be followed by training and fine tuning.

### Test 1: Testing if Mistral can recognize package emails
- I exported 234 emails via Google Takeout from my personal inbox. 
- These emails were split in two labels: “package” (with package related content from 3 carriers: PostNL, DHL and UPS) and “not-package”.
- This was a simple test, in which I asked the model to simply return a JSON reply with `{"is_package": true} or {"is_package": false}`

- After running the testing script, I’ve got the following result:
  - 129/134 package emails recognized, 99/100 non-package emails correctly rejected
  - 97.4% overall, about 2 seconds per email
- 6 mistakes: 
  - 4 cases were actually a single thread, in which the model was actually correct: it was a discussion about a sold out item marked as “package” (the model said `is_package: false`). These items were re-labeled as “not-package”
  - In 1 case, the model marked a package email as false because of the lack of a tracking number. This led to adjustments in the script’s prompt.
  - In 1 case, it was a spam email from AliExpress that the model marked as a packageThis also led to adjustments in the prompt.

### Test 2: Adding more complex features to be tested by the model
- Following the first test, I introduced new features to be tested by the model. This time, besides identifying if it’s a package, I also asked the model to label the email type in the following categories:  "upcoming", “delivered", "pickup",  "return" and "pending" (or null in `{"is_package": false}` cases)
- I increased the number of emails to 431. DPD emails were also added to the “package” labeled emails. 
- These emails had a new categorization: 
  - “package” now has sub-labels for each service: “postnl”, “dhl”, “ups”, “dpd”.
  - “not-package”: I added some gotchas here, such as emails from carriers with promotions or quality surveys.
- I had to tweak the script and the prompt a few times to include and filter some exceptional cases:
  - Customer satisfaction surveys.
  - A delivery confirmation AND customer satisfaction survey which made the model confused.
  - An email with a delivery code, but scrambled tracking code (for security reasons) which made the model reject it.
  - With the latest prompt and some fixes in mismatching labels, only 1/431 messages was miscategorized (a fake delivery ad by AliExpress). I decided to treat it later with filters.
  - **Script**: [evaluate.py](https://github.com/heloisabiagi/jijikat-delivery-tracker/blob/main/llm/scripts/evaluate.py)

### Test 3: Extracting the Tracking Number
- In this round, I tested if the model could correctly extract the tracking number by comparing it with the result of a regular expression.
- PostNL, DHL and UPS: easy cases in which the model picked the tracking number correctly in almost 100% of the cases
  - Their tracking number follows a clear, identifiable pattern
  - They always have the tracking number in the message body
  - It was easy to create a Regex pattern to capture the tracking number, as well as the model didn’t have difficulties to match the number captured by the Regex
  - Only PostNL had a tricky situation: in one email with a receipt code (ontvangstcode) for the courier, the tracking number was masked and the model couldn’t predict it correctly.
- DPD: good case study, because:
  - Not all of the emails have a tracking number in the body.
  - The tracking number is a sequence of 14 digits, with some gotchas: it can be followed by a Pakketreferentie, which is also 14 digits, and a single email can list several tracking numbers. These made it harder to write a rule to check the model’s answers. The model itself handled both cases well.
  - The most interesting behavior came when there was no tracking number: the model struggled to admit it, picking random numbers or even strings (like the recipient’s name) instead of simply saying “none”.
  - DPD Results:
    - When the email contained a tracking number (64 emails), the model found it every time (61 exact, 3 with extra junk attached).
    - In the 34 emails with no tracking number, it forced a value in 33.
    - Overall: 62/98 for the model alone, and 98/98 after a simple format check that keeps only 14-digit answers.
    - The correct answers were generated by a rule (“pakketnummer(s)”, skipping “Pakketreferentie”) and spot-checked in Gmail.
    - **Script**: [extract.py](https://github.com/heloisabiagi/jijikat-delivery-tracker/blob/main/llm/scripts/extract.py)

  - Comparing how carriers handle tracking numbers:

| | PostNL | DHL | UPS | DPD |
|---|---|---|---|---|
| **Number format** | `3S…` | `JVGL…`, `JJD…`, `3S…`, international `XX123456789DE` | `1Z…` | 14 digits |
| **Number in subject line** | sometimes | usually | not checked | no |
| **Parcels per email** | always 1 (a split order = separate emails) | 1 | 1 | sometimes several ("pakketnummer(s):") |
| **Emails without a number** | none in sample (one masked code) | none in sample | none in sample | ~35% (34 of 98) |
| **Look-alike numbers** | ontvangstcode (receipt code) | shop references | none seen | 14-digit Pakketreferentie |
