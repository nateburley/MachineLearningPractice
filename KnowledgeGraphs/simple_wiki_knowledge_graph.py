"""
This file mines Wikipedia articles for syntax and keywords, and builds a knowledge graph.

Author: Nathaniel M. Burley

Notes:
- Explore if vector embeddings of words can be converted to knowledge graphs, and vice 
  versa.

Sources:
    - https://towardsdatascience.com/auto-generated-knowledge-graphs-92ca99a81121
    - Really useful overview, with real-world examples and lots of pictures (for class paper):
        https://usc-isi-i2.github.io/slides/2018-02-aaai-tutorial-constructing-kgs.pdf
    
"""

# Wikipedia scraping libraries
import wikipediaapi  # pip install wikipedia-api
import pandas as pd
import concurrent.futures
from tqdm import tqdm
# NLP/Computational Linguistics libraries
import re
import spacy
import neuralcoref
# Graph libraries
import networkx as nx
import matplotlib.pyplot as plt
# Miscellaneous
import pickle



########################################################################################################################
## SCRAPE WIKIPEDIA REGARDING A TOPIC
########################################################################################################################

# Function that gets all unique links in an array of pages
def getPageLinks(page_names):
    page_links = []
    for page_name in page_names:
        new_links = list(page_name.links.keys())
        page_links = list(set().union(page_links, new_links))
    
    print("\n\nLINKS: {}".format(page_links))
    return page_links

# Function that scrapes Wikipedia for a given topic
def wikiScrape(topic_names, verbose=True):
    def wikiLink(link):
        try:
            page = wiki_api.page(link)
            if page.exists():
                return {'page': link, 'text': page.text, 'link': page.fullurl, 'categories': list(page.categories.keys())}
        except:
            return None

    wiki_api = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    page_names = [wiki_api.page(topic_name) for topic_name in topic_names]
    # Remove all NULL/invalid pages
    for page_name in page_names:
        index = page_names.index(page_name)
        if not page_name.exists():
            print('Page {} does not exist.'.format(topic_names[index]))
            page_names.remove(page_name)
            topic_names.remove()
        else:
            print(page_name)
    
    #page_links = list(page_name.links.keys())
    page_links = getPageLinks(page_names)
    progress = tqdm(desc='Links Scraped', unit='', total=len(page_links)) if verbose else None
    sources = [{'page': topic_name, 'text': page_name.text, 'link': page_name.fullurl, \
        'categories': list(page_name.categories.keys())} for topic_name, page_name in zip(topic_names, page_names)]
    
    # Parallelize the scraping, to speed it up (?)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_link = {executor.submit(wikiLink, link): link for link in page_links}
        for future in concurrent.futures.as_completed(future_link):
            data = future.result()
            sources.append(data) if data else None
            progress.update(1) if verbose else None     
    progress.close() if verbose else None
    
    namespaces = ('Wikipedia', 'Special', 'Talk', 'LyricWiki', 'File', 'MediaWiki', 'Template', 'Help', 'User', \
        'Category talk', 'Portal talk')
    sources = pd.DataFrame(sources)
    sources = sources[(len(sources['text']) > 20) & ~(sources['page'].str.startswith(namespaces, na=True))]
    sources['categories'] = sources.categories.apply(lambda x: [y[9:] for y in x])
    #sources['topic'] = topic_name # Seems redundant? Since 'page' is already set to topic_name?
    print('Wikipedia pages scraped:', len(sources))
    
    return sources



########################################################################################################################
## COMPUTATIONAL LINGUISTICS/NLP FUNCTIONING
# This section contains functions that do dependency parsing and find subject-predicate-object dependencies
# TODO: Look into coreference resolution to remove redundancies, normalize, etc. [adds a neural net here]
"""
NOTES:
    - Further reading, and helpful information:
        - https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/
        - https://www.analyticsvidhya.com/blog/2020/06/nlp-project-information-extraction/
        - https://www.analyticsvidhya.com/blog/2019/09/introduction-information-extraction-python-spacy/
        - https://realpython.com/natural-language-processing-spacy-python/
        - https://programmerbackpack.com/python-knowledge-graph-understanding-semantic-relationships/

    - Word2Vec similarity could be used to drop irrelevant terms
        - Train the algorithm on all the pages used and then drop terms with low co-occurences (?)
          That might only work for large datasets, but...could be worth looking into!

    - Honestly, I should probably re-write this from scratch using better logic and sentence structure
      for the relationship tuples. I can hand-code lots of simple rules, and maybe find libraries with more

    * Something else that could be an interesting side project, perhaps on its own: inferring semantic rules from similar
      dependency trees! The article (linked below) gives an introduction, but it would be interesting to put some thought
      into mathematically determining how similar given trees are...this could actually even involve some CBR:
        - Take some common trees (hyper/hyponyms, etc.) and make rules for them
        - Next, for a sentence that doesn't fit into a pre-defined rule, figure out the most similar dependency tree, and
          modify the rule as needed to fit the new sentence!
        - That rule then goes into our list of rules, and the iteration proceeds
      This could be a frickin' paper. "Automatic Semantic Rule Inference Using Case Based Reasoning"
    - On that note, here's an algorithm for comparing trees: https://arxiv.org/pdf/1508.03381.pdf
"""
########################################################################################################################

#nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)

# spacy.util.filter_spans
def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    # For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result

# Function that gets pairs of subjects and relationships
def getEntityPairs(text, coref=True):
    # preprocess text
    text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
    text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
    text = nlp(text)
    if coref:
        text = nlp(text._.coref_resolved)  # resolve coreference clusters

    def refineEntities(ent, sent):
        unwanted_tokens = (
            'PRON',  # pronouns
            'PART',  # particle
            'DET',  # determiner
            'SCONJ',  # subordinating conjunction
            'PUNCT',  # punctuation
            'SYM',  # symbol
            'X',  # other
        )
        ent_type = ent.ent_type_  # get entity type
        if ent_type == '':
            ent_type = 'NOUN_CHUNK'
            ent = ' '.join(str(t.text) for t in nlp(str(ent)) if t.pos_ not in unwanted_tokens and t.is_stop == False)

        elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
            refined = ''
            for i in range(len(sent) - ent.i):
                if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                    refined += ' ' + str(ent.nbor(i))
                else:
                    ent = refined.strip()
                    break

        return ent, ent_type

    sentences = [sent.string.strip() for sent in text.sents]  # split text into sentences
    ent_pairs = []
    for sent in sentences:
        sent = nlp(sent)
        spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
        spans = filter_spans(spans)
        with sent.retokenize() as retokenizer:
            [retokenizer.merge(span, attrs={'tag': span.root.tag, 'dep': span.root.dep}) for span in spans]
        deps = [token.dep_ for token in sent]

        # limit our example to simple sentences with one subject and object
        # if (deps.count('obj') + deps.count('dobj')) != 1 or (deps.count('subj') + deps.count('nsubj')) != 1:
            # continue

        for token in sent:
            if token.dep_ not in ('obj', 'dobj'):  # identify object nodes
                continue
            subject = [w for w in token.head.lefts if w.dep_ in ('subj', 'nsubj')]  # identify subject nodes
            if subject:
                subject = subject[0]
                # identify relationship by root dependency
                relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                if relation:
                    relation = relation[0]
                    # add adposition or particle to relationship
                    try:
                        if relation.nbor(1).pos_ in ('ADP', 'PART'):
                            relation = ' '.join((str(relation), str(relation.nbor(1))))
                    except:
                        print("Failed at line 207")
                        return
                else:
                    relation = 'unknown'

                subject, subject_type = refineEntities(subject, sent)
                token, object_type = refineEntities(token, sent)

                ent_pairs.append([str(subject), str(relation), str(token), str(subject_type), str(object_type)])

    ent_pairs = [sublist for sublist in ent_pairs if not any(str(ent) == '' for ent in sublist)]
    pairs = pd.DataFrame(ent_pairs, columns=['subject', 'relation', 'object', 'subject_type', 'object_type'])
    #print('Entity pairs extracted:', str(len(ent_pairs)))

    return pairs

# Function that extracts ALL the pairs. Not just the first one smh
def extractAllRelations(wiki_data):
    all_pairs = []
    for i in range(0, 100): #range(len(wiki_data.index)):
        #print(wiki_data.loc[i,'text'])
        pairs = getEntityPairs(wiki_data.loc[i,'text'])
        all_pairs.append(pairs)
        print("Made it through {} iterations".format(i))
    all_pairs_df = pd.concat(all_pairs)
    print("Successfully extracted {} entity pairs".format(len(all_pairs_df.index)))
    return all_pairs_df



########################################################################################################################
## FUNCTION THAT DRAWS, PLOTS THE KNOWLEDGE GRAPH
########################################################################################################################

# Function that plots and draws a knowledge graph
def drawKG(pairs):
    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object', create_using=nx.MultiDiGraph())
    node_deg = nx.degree(k_graph)
    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(120, 90), dpi=80)
    nx.draw_networkx(
        k_graph,
        node_size=[int(deg[1]) * 1000 for deg in node_deg],
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='red',
        edgecolors='black',
        node_color='green',
        )
    labels = dict(zip(list(zip(pairs.subject, pairs.object)), pairs['relation'].tolist()))
    print(labels)
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels, font_color='black')
    plt.axis('off')
    plt.show()
    plt.savefig('church_knowledge_graph.png')


# Function that plots a "subgraph", if the main graph is too messy
# Example use: filterGraph(pairs, 'Directed graphs')
def filterGraph(pairs, node):
    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object', create_using=nx.MultiDiGraph())
    edges = nx.dfs_successors(k_graph, node)
    nodes = []
    for k, v in edges.items():
        nodes.extend([k])
        nodes.extend(v)
    subgraph = k_graph.subgraph(nodes)
    layout = (nx.random_layout(k_graph))
    nx.draw_networkx(
        subgraph,
        node_size=1000,
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='red',
        edgecolors='black',
        node_color='white'
        )
    labels = dict(zip((list(zip(pairs.subject, pairs.object))), pairs['relation'].tolist()))
    edges= tuple(subgraph.out_edges(data=False))
    sublabels ={k: labels[k] for k in edges}
    nx.draw_networkx_edge_labels(subgraph, pos=layout, edge_labels=sublabels, font_color='red')
    plt.axis('off')
    plt.show()
    plt.savefig('church_knowledge_graph.png')



########################################################################################################################
## DO THE THING: SCRAPE WIKIPEDIA, APPLY NLP, AND BUILD A KNOWLEDGE GRAPH
# TODO: Lots of random pages that aren't that relevant get scraped. Can those nodes be dropped later, since in theory
#       they should be "less connected" or "weaker connected" than the other nodes that are actually "on topic"?
########################################################################################################################

# Scrape Wikipedia for a topic
wiki_data = wikiScrape(['Catholic Church', 'Islam', 'Russian Orthodox Church', 'Judaism', 'Buddhism', 'Panpsychism', 'UFO religion'])
print("WIKIPEDIA SCRAPE DF LENGTH: {}".format(len(wiki_data.index)))
print(wiki_data.head(25))
print("\n")

# Pickle the wiki_data to not have to scrape a million times
datafile = open('religion_wiki_data', 'wb')
pickle.dump(wiki_data, datafile)

# Load in the wiki data, so we don't have to scrape
# infile = open('religion_wiki_data', 'rb')
# wiki_data = pickle.load(infile)

# Get subject object relationships (which form vertices and edges in the graph)
# TODO: Parallelize (thread) the shit out of this
#   - https://realpython.com/intro-to-python-threading/
all_pairs = extractAllRelations(wiki_data)
print("ENTITY PAIRS-- SUBJECT/OBJECT RELATIONSHIPS LENGTH: {}".format(len(all_pairs.index)))
print(all_pairs.head(20))
print(all_pairs.tail(20))
print("\n")

# Draw the graph
drawKG(all_pairs)
