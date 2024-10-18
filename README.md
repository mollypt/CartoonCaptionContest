 Cartoons in the New Yorker Cartoon Caption Contest are usually funny because of their incongruity—they juxtapose the mundane and the absurd. The mundane element of an illustration often references some recognizable facet of readers' lives, creating a shared sense of identity among readers, who, by submitting captions, show they're in on the joke, writes Philip Smith. 

If The New Yorker represents the lives of its readers, then how are its readers represented? 
For my junior independent work, I used machine learning to explore themes in the New Yorker Cartoon 
Caption Contest, with focus on identifying the most common settings and characters. 

My project had the following components:

1.) Use the GPT-4 API to generate setting and character labels for ~700 cartoons

2.) Obtain word embeddings (vector representations) of all labels 

3.) Perform k-means clustering on the settings and character embeddings to identify the most common setting and character categories

4.) Evaluate clustering results by calculating loss and silhouette scores

Of the ~700 cartoons considered, domestic settings were most common, with the Living Room, Bedroom/Kitchen, 
and Front Yard clusters spanning 103 cartoons. The corporate workplace was also highly represented, with the Office 
and Meeting Room clusters containing a combined 86 cartoons. Clustering also identified several recurring settings,
like The Doctor's Office (19) and the Therapist's Office (18), with wackier clusters being Ocean/Island/Mountains (24),
Heaven/Hell (20), Cave (13), and Desert (16).

The most common character by far is the Businessman, corresponding to a cluster of 268 characters. The second- and third- largest
clusters are the more generic Woman (205) and Man (166), followed by Wild Animal/Creature with 150 characters (such as sea monster, 
dragon, and woolly mammoth). There are 44 Cavemen across the cartoon collection, and 44 characters within the Angel/Devil/Alien cluster. 

The anomaly clusters were far less intelligible, but they were still interesting. In at least 13 cartoons, businessmen were 
the anomaly—businessman were presented out of context, for example, at the beach. In another cluster of 45 cartoons, the 
anomaly was some oversized object, oftentimes pertaining to the domestic sphere (e.g. "giant vacuum cleaner, giant table 
with food on it"). These incongruities seem to make fun of domestic life by drawing absurdity from the mundane. 
