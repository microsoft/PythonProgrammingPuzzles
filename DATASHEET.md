# Dataset datasheet
Here we provide a dataset *datasheet* as defined by:

T. Gebru, J.Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, and K. Crawford. [Datasheets for Datasets](https://arxiv.org/abs/1803.09010). 


### Motivation
* For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.
*The dataset is created to evaluate and possibly teach machines to program in Python.*
* Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? *The dataset was
initiated by Microsoft researchers and interns, primarily Adam Tauman Kalai and Tal Schuster, but we hope many other
people will contribute.*
* Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number. *Microsoft Research*
* Any other comments?
### Composition
* What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.
*The dataset consists of programming puzzles. Each puzzle is a short snippet (often one-line) of Python code defining a function. To solve the puzzle, 
one must find an input that makes the function return `True`*
* How many instances are there in total (of each type, if appropriate)? *The datasest is intended to grow! At the time of creation it consisted of 139,072 puzzle instances derived from 200 problem templates.*
* Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable). *N/A*
* What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description. *The dataset has .json files with Python snippets. There is also code provided to generate such files (or larger ones if desired) and work with the puzzles.*
* Is there a label or target associated with each instance? If so, please provide a description. *At the time of creation, for convenience (and debugging of the puzzle) over 90\% of the puzzles had one or more solutions. However, puzzles need not have any reference solution.*
* Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text. *N/A*
* Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)? If so, please describe how these relationships are made explicit. *The problem template that generated each puzzle is apparent in its `name`* 
* Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them. *No recommended splits have been provided. However the study puzzles serve as a reasonable benchmark.*
* Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description. *All known errors should be fixed or the puzzles removed. The history will be apparent in the GitHub commits.*
* Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)? If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate. *Self-contained.*
* Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor–patient confidentiality, data that includes the content of individuals’ non-public communications)? If so, please provide a description. *No*
* Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why. *No*
* Does the dataset relate to people? If not, you may skip the remaining questions in this section. *No*
* Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions
* Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how.
* Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description.
* Any other comments?
### Collection Process
* How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly 
inferred/derived from other data, was the data validated/verified? If so, please describe how. *Initial data was collected from: Wikipedia, codeforces.com, and ICPC and IMO problems. These problems were converted as discussed in the paper. Further puzzles may be contributed from these sources, other sources, or the creators imagination.*
* What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated? *Problems are created manually, puzzles instances are generated by running the puzzle generators written in each problem template.*
* If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)? *For puzzles inspired by codeforces problems, the problems were added in order of number of people who solved the problems.* 
* Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)? *The authors of the paper contributed the initial problems.*
* Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created. *The problems were created during 2020-2021.*
* Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation. *A user study was run on the `study` problems. See the arXiv paper. However, since nothing about the user
study is included in this repo, there is no need to reference the IRB.* 
* Does the dataset relate to people? If not, you may skip the remaining questions in this section. *No*
* Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?
Gebru et al. Datasheets for Datasets 4
* Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.
* Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.
* If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).
* Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.
* Any other comments?
### Preprocessing/cleaning/labeling
* Was any preprocessing/cleaning/labeling of the data done 
(e.g., discretization or bucketing, tokenization, 
part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? 
If so, please provide a description. If not, you may skip the remainder of the questions in this section. *No*
* Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data.
* Is the software used to preprocess/clean/label the instances available? If so, please provide a link or other access point.
* Any other comments?
### Uses
* Has the dataset been used for any tasks already? If so, please provide a description. *See upcoming arXiv paper.*
* Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point. *Relevant papers will be linked to in GitHub*
* What (other) tasks could the dataset be used for? *Dataset could be used to teach or evaluate human programmers in Python as well as computers*
* Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled
that might impact future uses? For example, is there anything that a future user might need to know to avoid uses that 
could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other
undesirable harms (e.g., financial harms, legal risks) If so, please provide a description. Is there anything a 
future user could do to mitigate these undesirable harms? *N/A*
* Are there tasks for which the dataset should not be used? If so, please provide a description. 
*The solutions to the problems only test the given puzzle instances, they do not guarantee worst-case correctness of
an algorithm.*
* Any other comments? 
### Distribution
* Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? *GitHub*
* How will the dataset will be distributed (e.g., tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)? *GitHub*
* When will the dataset be distributed? The dataset was initially released in June, 2021.   
* Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions. *Since the dataset consists of code, it is released under an MIT license*
* Have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions. *Problems should have a link back to the source, e.g., problems inspired by codeforces problems have links back to the original problem.*
* Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation. *N/A*
### Maintenance
* Who will be supporting/hosting/maintaining the dataset? *The dataset is an open-source project hosted at github.*
* How can the owner/curator/manager of the dataset be contacted (e.g., email address)? *Communication should happen through GitHub*
* Is there an erratum? *Errors may be reported through GitHub issues and/or corrected through a pull request.*
* Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? *GitHub*
* If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)? *N/A:
 No personal data* 
* Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to users. *Older versions will remain on GitHub for reproducibility* 
* If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? *Contributions are welcome and appreciated through GitHub*
* Any other comments?
