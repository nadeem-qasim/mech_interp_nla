# Claim spot-check — BLIND (label every item before opening the KEY)

For write `human_label: true`, `false` or `irrelevant`, using the passage prefix and fve_claims/ANNOTATION_GUIDE.md  
true = stated or entailed by the passage prefix; false = contradicted by passage prefix  
irrelevant = forecasts / what comes next. `note:` is optional. Random and largest-drop items are mixed

## 1. S01

Model 27B · document 35

**Passage:** Frederick August Wenderoth or F. A. Wenderoth (1819 – 1884) was a German-born American painter and photographer. Born and educated in Cassel, where he first learned to paint from his father, he established a lifelong friendship with Charles Christian Nahl at school. During a period of political upheaval he left Germany for Paris where his was joined by Nahl and his half-brother Hugo Wilhelm Arthur Nahl. They then moved to the US, living first in New York, before traveling by sea to California to join the Gold Rush. Unsuccessful as miners, Wenderoth and Nahl opened art studios, first in Sacramento and later in San Francisco, collaborating as painters, engravers and photographers. After a trip to South Seas and Australia, Wenderoth married and moved to Philadelphia, where he established a photography studio. In the late 1850s he worked for a period in South Carolina, going into partnership with Jesse Bolles. There, and later when he returned to Philadelphia, he innovated a number of photographic techniques, such as the ivory-type and photozincography. Wenderoth died in 1884 of tuberculosis

**AV wrote:** "of tuberculosis … of tuberculosis"

**Claim:** He died of tuberculosis.

human_label: **true**

note: entire passage was the passed prefix, easy claim / last sentence

## 2. S02

Model 7B · document 19

**Passage:** "Lisa Mona Lisa" was the Austrian entry in the Eurovision Song Contest 1988, performed in German by Wilfried. The song was performed twelfth on the night, following Germany's Maxi & Chris Garden with "Lied für einen Freund" and preceding Denmark's Kirsten & Søren with "Ka 'du se hva' jeg sa'?". At the close of voting, it had received no points, placing it last out of a field of 21. The song is a ballad in which the singer likens the object of his affection to the Mona Lisa, in that she is very mysterious but nonetheless entrancing. The song was recorded in German, French and English. Due to the song's poor performance, the English version was not released. It was succeeded as Austrian representative at the 1989 Contest by Thomas Forstner singing

**AV wrote:** "Eurovision Song Contest 1985"

**Claim:** The described Eurovision entry was in 1985.

human_label: **false**

note: 

## 3. S03

Model 27B · document 33

**Passage:** Alan Belford Jones AO (born 13 April 1941, or possibly 1942 or 1943) is an Australian radio broadcaster. He is a former coach of the Australia national rugby union team and rugby league coach and administrator. He has worked as a school teacher, a speech writer in the office of the Prime Minister Malcolm Fraser, and in musical theatre. He has a Bachelor of Arts from the University of Queensland, and completed a one-year teaching diploma at Worcester College, Oxford. He has received civil and industry awards. Jones hosts a popular Sydney breakfast radio program, on radio station 2 GB. Jones advocates mainly conservative views, and the popularity of his radio program has made him a highly paid and influential media personality in Australia. Despite his success, he remains a controversial figure. His on-air conduct has received adverse findings from Australia's media regulators, and he has frequently been sued for defamation

**AV wrote:** "has been sued frequently for defamation … has been sued frequently for defamation … has been sued frequently for defamation"

**Claim:** He has been sued frequently for defamation.

human_label: **true** 

note: easy claim / last sentence

## 4. S04

Model 7B · document 55

**Passage:** Central Massachusetts is the geographically central region of Massachusetts. Though definitions vary, most include all of Worcester County and the northwest corner of Middlesex County. Worcester, the largest city in the area and the seat of Worcester County, is often considered the cultural capital of the region. Other populous cities include Fitchburg, Gardner, Leominster, and arguably Marlborough. The region is mostly wooded and hilly upland, in contrast to the Atlantic coastal plain to the east and the Connecticut River valley lowland to the west. The geographic center of Massachusetts is located in the town of Rutland. The term is seldom used in eastern Massachusetts, where Worcester and points west are instead regarded as part of Western Mass.

**AV wrote:** "the term 'Middle Pennsylvania' is used to distinguish the region from surrounding areas"

**Claim:** Middle Pennsylvania is used in some contexts to distinguish the region.

human_label: **false**

note: total invention, PA not mentioned

## 5. S05

Model 27B · document 20

**Passage:** The Church of Saint Francis of Assisi is a Rococo Catholic church in Ouro Preto, Brazil. Its erection began in 1766 after a design by the Brazilian architect and sculptor Antônio Francisco Lisboa, otherwise known as Aleijadinho. Lisboa designed both the structure of the church and the carved decorations on the interior, which were only finished towards the end of the 19th century. The circular bell towers and the oculus closed by a relief were original features in religious architecture of that time in Brazil. The façade has a single entrance door under a soapstone frontispiece under a relief depicting Saint Francis receiving the stigmata. The interior is richly decorated with golden woodwork, statues and paintings, and the wooden ceiling displays a painting by Manuel da Costa Ataíde. Due to its architecture and historical significance regarding eighteenth century gold mining, this church is classified on the UNESCO World Heritage List

**AV wrote:** "it is inscribed on the UNESCO World Heritage List … it is inscribed on the UNESCO World Heritage List"

**Claim:** The church is inscribed on the UNESCO World Heritage List.

human_label: **false**

note: close but wrong verb; ++EDIT: was wrong, "inscribed" is UNESCO's standard term++

## 6. S06

Model 7B · document 51

**Passage:** Jefferson Madeira da Silva, or simply Jefferson Madeira (born 15 February 1988) is a Brazilian football forward playing with Duque de Caxias Born in Duque de Caxias, Madeira started his career playing with Sport Recife in 2007 making his debut in the Campeonato Brasileiro Série A. After loan spells with Metropolitano and Treze during 2008, he joined on January 2009 FK Teleoptik-a club owned by FK Partizan Belgrade-playing in the 2008 – 09 Serbian League Belgrade playing along compatriots Washington Santana da Silva, Alex dos Santos Gonçalves and Elton Martins, helping Teleoptik to finish 2nd in the 2008 – 09 Serbian League Belgrade, one of Serbian 3rd tiers, thus earning promotion to the 2009 – 10 Serbian First League. In summer 2009 he returned to Brazil and played until the end of the year with Olaria, before returning to Sport Recife

**AV wrote:** "he returned to Serie A again"

**Claim:** Anderson Sales returned to Serie A.

human_label: **false**

note: 

## 7. S07

Model 27B · document 18

**Passage:** MCPON John-117, or "Master Chief", is a fictional character and the protagonist of the Halo fictional universe created by Bungie. Master Chief is a playable character in the series of science fiction first-person shooter video games, Halo 2, Halo 3, Halo 4, and. Outside video games, the character appears in books and graphic novels—,,, and —and has cameos in other Halo media, including,, The Halo Graphic Novel, Halo Legends and. The Master Chief is a towering biochemically and cybernetically-enhanced supersoldier, raised and trained from an early age to be a weapon; he is almost faceless, never seen without his green-colored armor and helmet. He is commonly referred to by his naval rank, rather than his given birth-name or serial-designation. The character is voiced by Steve Downes, a Chicago disc jockey, in the video games in which he appears. Downes based his personification of the Chief on an initial character sketch which called for a Clint Eastwood – type character of few words. The Master Chief is a video game icon, a relative newcomer among more established franchise characters, such as Mario, Sonic the Hedgehog and Lara Croft. The character has received a mostly positive reception. While some have described the Chief's silent and faceless nature as a weakness of the character, other publications have suggested these attributes better allow players to assume his role

**AV wrote:** "better allows them to assume his role … better allows players to assume his role … better allows players to assume his role"

**Claim:** The character's anonymity/attributes better allow players to assume his role.

human_label: **true**

note: easy / last token

## 8. S08

Model 27B · document 69

**Passage:** Appenzell is a village, a statistic town, but not a municipality, and the capital of the canton of Appenzell Innerrhoden in Switzerland. Appenzell has no municipal government of its own; rather, the different parts of Appenzell belong to and are governed by the districts (and municipalities) Appenzell, Schwende and Rüte. Because of that, for firefighting, energy and water, the village Appenzell has a special-purpose municipality, the Feuerschaugemeinde. In 1071 the village was referred to as Abbacella. By 1223 this changed to Abbatiscella, meaning the Abbot's cell. This refers to the abbot of the Abbey of Saint Gall. The official language of Appenzell is (the Swiss variety of Standard) German, but the main spoken language is the local variant of the Alemannic Swiss German dialect. The buildings in the village core, the parish church, the 1563 town hall, the Salesis house, the ruins of Castle Clanx and the state archives with the administration building are listed as heritage sites of national significance

**AV wrote:** "national significance … national significance … national significance"

**Claim:** The buildings are listed as heritage sites of national significance.

human_label: **true**

note: easy / last token

## 9. S09

Model 7B · document 18

**Passage:** MCPON John-117, or "Master Chief", is a fictional character and the protagonist of the Halo fictional universe created by Bungie. Master Chief is a playable character in the series of science fiction first-person shooter video games, Halo 2, Halo 3, Halo 4, and. Outside video games, the character appears in books and graphic novels—,,, and —and has cameos in other Halo media, including,, The Halo Graphic Novel, Halo Legends and. The Master Chief is a towering biochemically and cybernetically-enhanced supersoldier, raised and trained from an early age to be a weapon; he is almost faceless, never seen without his green-colored armor and helmet. He is commonly referred to by his naval rank, rather than his given birth-name or serial-designation. The character is voiced by Steve Downes, a Chicago disc jockey, in the video games in which he appears. Downes based his personification of the Chief on an initial character sketch which called for a Clint Eastwood – type character of few words. The Master Chief is a video game icon, a relative newcomer among more established franchise characters, such as Mario, Sonic the Hedgehog and Lara Croft. The character has received a mostly positive reception. While some have described the Chief's silent and faceless nature as a weakness of the character, other publications have suggested these attributes better allow players to assume his role

**AV wrote:** "Final token "role""

**Claim:** The final tokenizer token is “role”.

human_label: **true**

note: easy / last token

## 10. S10

Model 27B · document 44

**Passage:** I, Frankenstein is a 2014 Australian-American action-horror film written and directed by Stuart Beattie, based on the digital-only graphic novel by Kevin Grevioux. The film was produced by Tom Rosenberg, Gary Lucchesi, Richard Wright, Andrew Mason and Sidney Kimmel. It stars Aaron Eckhart, Bill Nighy, Yvonne Strahovski, Miranda Otto and Jai Courtney. The film brings the story of Adam, Frankenstein's monster, going on a dangerous journey and determined to stop evil demons and their ruthless leader from taking over the world. The film was released on January 24, 2014, in the United States and on March 20, 2014 in Australia. The film grossed $71 million worldwide against production budget of $65 million

**AV wrote:** "closes the box office figure"

**Claim:** The final token closes the box office figure.

human_label: **true** 

note: borderline but I'd mark true, "closes"

## 11. S11

Model 7B · document 90

**Passage:** Henri de Buade de Frontenac (1596 – 1622) was a French aristocrat during the age of Louis XIII of France, best known as the father of Louis de Buade de Frontenac, the future Lieutenant General of the colony of New France in North America. Henri de Buade de Frontenac was born in 1596, son of Antoine de Buade and Anne de Secondat. His father, from a family that originated in Guyenne, was an intimate of King Henry IV of France As a child Henri de Buade was a playmate of the future king Louis XIII. It is said that one day when King Henri IV was in poor health, he had the two boys stage a fight on his bed to amuse him. In May 1612 King Louis XIII granted him some land behind the Château du Louvre in Paris, then used only for a hen house, on which he could build a house. His father Antoine arranged for Henri to marry Anne Phélypeaux in 1613. Her father and uncle were Raymond Phélypeaux and Paul Phélypeaux, both secretaries of state and highly influential men. His son, Louis de Buade, Compte de Frontenac at de Pulluau, was born in 1620. King Louis XIII acted as godfather to the boy, who was named after him. Henri de Buade became a colonel in the Regiment of Navarre. He was killed in 1622 during a military campaign. His heart was removed, sealed in a lead box, and buried in the church at Palluau. Henri's son Louis later became Lieutenant General of the colony of New France in North America

**AV wrote:** "Final token "America""

**Claim:** The final token is "America".

human_label: **true**

note: easy / last token

## 12. S12

Model 27B · document 56

**Passage:** The Atlantic Charter was a pivotal policy statement issued during World War II on 14 August 1941 which defined the Allied goals for the post-war world. The leaders of the United Kingdom and the United States drafted the work and all the Allies of World War II later confirmed it. The Charter stated the ideal goals of the war: no territorial aggrandizement; no territorial changes made against the wishes of the people (self-determination); restoration of self-government to those deprived of it; reduction of trade restrictions; global cooperation to secure better economic and social conditions for all; freedom from fear and want; freedom of the seas; and abandonment of the use of force, as well as disarmament of aggressor nations. Adherents of the Atlantic Charter signed the Declaration by United Nations on 1 January 1942, which became the basis for the modern United Nations. The Atlantic Charter set goals for the postwar world and inspired many of the international agreements that followed the war. The General Agreement on Tariffs and Trade (GATT), the postwar independence of European colonies, and much more are derived from the Atlantic Charter

**AV wrote:** "it inspired many agreements and policies"

**Claim:** The Atlantic Charter inspired many agreements and policies.

human_label: **true** 

note: 

## 13. S13

Model 27B · document 84

**Passage:** Jared Scot Allen (born April 3, 1982) is a former professional American football defensive end. After playing college football for Idaho State University, he was drafted by the Kansas City Chiefs of the National Football League (NFL) in the fourth round of the 2004 NFL Draft. After four years with the Chiefs, Allen was traded to the Minnesota Vikings in exchange for three draft picks, including a first-rounder. He spent six years with the Vikings before joining the Chicago Bears as a free agent in 2014. During the 2015 season, he was traded to the Carolina Panthers, with whom he played in his only career Super Bowl. A five-time Pro Bowl and four-time All-Pro selection, Allen tallied 136 quarterback sacks during his 12-year career

**AV wrote:** "The sentence lists career accomplishments in sequence — Pro Bowls, All-Pro selections, Super Bowl appearances"

**Claim:** The sentence lists career accomplishments in sequence — Pro Bowls, All-Pro selections, Super Bowl appearances.

human_label: **false** 

note: ***borderline,*** details true, simply not one sentence but last two++; mislabeled as **true** by Claude++

## 14. S14

Model 27B · document 43

**Passage:** The Pantabangan – Carranglan Watershed Forest Reserve is a conservation area located in the upper reaches of the Pampanga River in Nueva Ecija, Philippines, and borders the Sierra Madre and Caraballo Mountains in Aurora and Nueva Vizcaya. It encompasses of the drainage basin surrounding the Pantabangan Lake, an impoundment of the Pampanga River by the Pantabangan Dam. The multi-purpose dam is situated at the confluence of Pampanga River's two headwaters, namely the Pantabangan and Carranglan Rivers in the municipality of Pantabangan. It stretches above the dam site for to where Carranglan River originates in the Caraballo on the north, and for to where Pantabangan River originates in the Sierra Madre on the east. It is considered a critical watershed for the agricultural economy and hydroelectric power generation in the region of Central Luzon

**AV wrote:** "Final token "Yucatón""

**Claim:** The final token is "Yucatón".

human_label: **false** 

note: 

## 15. S15

Model 7B · document 54

**Passage:** American Theocracy: The Peril and Politics of Radical Religion, Oil, and Borrowed Money in the 21st Century () is a 2005 political commentary book by American political writer Kevin Phillips. The book is a critique of the past forty years of the Republican coalition in United States politics. He "presents a nightmarish vision of ideological extremism, catastrophic fiscal irresponsibility, rampant greed, and dangerous shortsightedness." Phillips points to three unifying themes holding this coalition together. First, its tie to oil and the role oil plays in American and world events. Second, to the coalition of social conservatives, Evangelicals and Pentecostals in this Republican coalition. Finally, he points to the "debt culture" of this coalition, and to a coming "debt bubble" related to the debt of the U.S. Government and U.S. consumers. He argues that similar issues have been prevalent in the past, when other world powers, such as the Roman Empire and the British Empire declined from their peaks and fell into disarray. While working as a strategist in the presidential campaign of Richard Nixon, Phillips wrote The Emerging Republican Majority. In that book, Phillips predicted the formation of this very coalition that he criticizes in his current book. In American Theocracy he admits that while these "mutations," as he calls them, could have been predicted, he did not foresee the extent to which they would develop and dominate the coalition he helped put together. The last chapter of this book references his first work, and is called

**AV wrote:** "referencing the earlier book's theme"

**Claim:** The final chapter references the earlier book.

human_label: **true**

note: 

## 16. S16

Model 27B · document 42

**Passage:** Pedro León Díaz Gallo (29 June 1782 – 7 February 1852) was an Argentine statesman and priest. He was a representative to the Congress of Tucumán which on 9 July 1816 declared the Independence of Argentina. Gallo was born in Santiago del Estero and studied at the Monserrat School in Córdoba until he was ordained, graduating as a teacher of art (or philosophy according to other sources) at the University of San Carlos. Gallo was elected to represent Santiago del Estero in the Tucumán Congress and served for the declaration in 1816. He was vice-president of the Congress in August 1816 and twice president after it was moved to Buenos Aires. When the Congress was dissolved in 1820, he and his colleagues were imprisoned as traitors. Gallo returned to Santiago del Estero and was a signatory of the peace treaty of Vinará in 1821, signing on his province's behalf with Pedro Miguel Aráoz of Tucumán and José Andrés Pacheco de Melo of Córdoba. He was a minister in the government of Juan Felipe Ibarra, before retiring in Tucumán where he died

**AV wrote:** "where he lived as a wealthy landowner"

**Claim:** He lived as a wealthy landowner in retirement.

human_label: **false** 

note: 

## 17. S17

Model 27B · document 46

**Passage:** Alice Bunker Stockham (November 8, 1833 in Cardington, Ohio – December 3, 1912 in Alhambra, California) was an obstetrician and gynecologist from Chicago and the fifth woman to become a doctor in the United States. She promoted gender equality, dress reform, birth control, and male and female sexual fulfillment for successful marriages. A well-traveled and well-read person who counted among her friends Leo Tolstoy and Havelock Ellis, she also visited Sweden and from her trips to schools there she brought back the idea of teaching children domestic crafts, thus single-handedly establishing shop and home economics classes in the United States. Stockham lectured against the use of corsets by women, made public endorsements of the healthiness of masturbation for both men and women (still controversial when echoed by U.S. Surgeon General Joycelyn Elders more than 100 years later), advocated complete abstinence from alcohol and tobacco, and believed in women's rights. Stockham was very concerned with the economic plight of divorced women with children and prostitutes who wanted to get off the street. She felt that these women had no marketable skills and would be unable to support themselves, so she had copies of her book Tokology, a layperson's guide to gynecology and midwifery, privately printed and gave them to "unfortunate women" to sell door-to-door in Chicago. Each copy came with a bound-in certificate signed by Stockham and entitling the bearer to a free gynecological exam. In 1905, a then 72-year old Stockham and her publisher were convicted of circulating improper literature under the Comstock laws

**AV wrote:** "the sentence is incomplete"

**Claim:** The sentence is incomplete.

human_label: **true** 

note: easy/last token; ++I was wrong,++ period's missing but that's by default we remove periods and activation on last token