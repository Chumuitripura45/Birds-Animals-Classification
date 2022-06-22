import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

### load file
map_dict = {0: 'AFRICAN CROWNED CRANE', 1: 'AFRICAN FIREFINCH', 2: 'ALBATROSS', 3: 'ALEXANDRINE PARAKEET',
       4: 'AMERICAN AVOCET', 5: 'AMERICAN BITTERN', 6: 'AMERICAN COOT', 7: 'AMERICAN GOLDFINCH',
       8: 'AMERICAN KESTREL', 9: 'AMERICAN PIPIT', 10: 'AMERICAN REDSTART', 11: 'ANHINGA',
       12: 'ANNAS HUMMINGBIRD', 13: 'ANTBIRD', 14: 'ARARIPE MANAKIN', 15: 'ASIAN CRESTED IBIS', 16: 'BALD EAGLE',
       17: 'BALI STARLING', 18: 'BALTIMORE ORIOLE', 19: 'BANANAQUIT', 20: 'BANDED BROADBILL', 21: 'BAR-TAILED GODWIT',
       22: 'BARN OWL', 23: 'BARN SWALLOW', 24: 'BARRED PUFFBIRD', 25: 'BAY-BREASTED WARBLER', 26: 'BEARDED BARBET',
       27: 'BEARDED REEDLING', 28: 'BELTED KINGFISHER', 29: 'BIRD OF PARADISE', 30: 'BLACK & YELLOW bROADBILL',
       31: 'BLACK FRANCOLIN', 32: 'BLACK SKIMMER', 33: 'BLACK SWAN', 34: 'BLACK TAIL CRAKE',
       35: 'BLACK THROATED BUSHTIT', 36: 'BLACK THROATED WARBLER', 37: 'BLACK VULTURE', 38: 'BLACK-CAPPED CHICKADEE',
       39: 'BLACK-NECKED GREBE', 40: 'BLACK-THROATED SPARROW', 41: 'BLACKBURNIAM WARBLER', 42: 'BLUE GROUSE',
       43: 'BLUE HERON', 44: 'BOBOLINK', 45: 'BORNEAN BRISTLEHEAD', 46: 'BORNEAN LEAFBIRD', 47: 'BROWN NOODY',
       48: 'BROWN THRASHER', 49: 'BULWERS PHEASANT', 50: 'CACTUS WREN', 51: 'CALIFORNIA CONDOR', 52: 'CALIFORNIA GULL',
       53: 'CALIFORNIA QUAIL', 54: 'CANARY', 55: 'CAPE MAY WARBLER', 56: 'CAPUCHINBIRD', 57: 'CARMINE BEE-EATER',
       58: 'CASPIAN TERN', 59: 'CASSOWARY', 60: 'CEDAR WAXWING', 61: 'CHARA DE COLLAR', 62: 'CHIPPING SPARROW',
       63: 'CHUKAR PARTRIDGE', 64: 'CINNAMON TEAL', 65: 'COCK OF THE  ROCK', 66: 'COCKATOO', 67: 'COMMON FIRECREST',
       68: 'COMMON GRACKLE', 69: 'COMMON HOUSE MARTIN', 70: 'COMMON LOON', 71: 'COMMON POORWILL', 72: 'COMMON STARLING',
       73: 'COUCHS KINGBIRD', 74: 'CRESTED AUKLET', 75: 'CRESTED CARACARA', 76: 'CRESTED NUTHATCH', 77: 'CROW',
       78: 'CROWNED PIGEON', 79: 'CUBAN TODY', 80: 'CURL CRESTED ARACURI', 81: 'D-ARNAUDS BARBET',
       82: 'DARK EYED JUNCO', 83: 'DOUBLE BARRED FINCH', 84: 'DOWNY WOODPECKER', 85: 'EASTERN BLUEBIRD',
       86: 'EASTERN MEADOWLARK', 87: 'EASTERN ROSELLA', 88: 'EASTERN TOWEE', 89: 'ELEGANT TROGON',
       90: 'ELLIOTS  PHEASANT', 91: 'EMPEROR PENGUIN', 92: 'EMU', 93: 'ENGGANO MYNA', 94: 'EURASIAN GOLDEN ORIOLE',
       95: 'EURASIAN MAGPIE', 96: 'EVENING GROSBEAK', 97: 'FIRE TAILLED MYZORNIS', 98: 'FLAME TANAGER', 99: 'FLAMINGO',
       100: 'FRIGATE', 101: 'GAMBELS QUAIL', 102: 'GANG GANG COCKATOO', 103: 'GILA WOODPECKER', 104: 'GILDED FLICKER',
       105: 'GLOSSY IBIS', 106: 'GO AWAY BIRD', 107: 'GOLD WING WARBLER', 108: 'GOLDEN CHEEKED WARBLER',
       109: 'GOLDEN CHLOROPHONIA', 110: 'GOLDEN EAGLE', 111: 'GOLDEN PHEASANT', 112: 'GOLDEN PIPIT',
       113: 'GOULDIAN FINCH', 114: 'GRAY CATBIRD', 115: 'GRAY PARTRIDGE', 116: 'GREAT POTOO',
       117: 'GREATOR SAGE GROUSE', 118: 'GREEN JAY', 119: 'GREEN MAGPIE', 120: 'GREY PLOVER',
       121: 'GUINEA TURACO', 122: 'GUINEAFOWL', 123: 'GYRFALCON', 124: 'HARPY EAGLE', 125: 'HAWAIIAN GOOSE',
       126: 'HELMET VANGA', 127: 'HIMALAYAN MONAL', 128: 'HOATZIN', 129: 'HOODED MERGANSER', 130: 'HOOPOES',
       131: 'HORNBILL', 132: 'HORNED GUAN', 133: 'HORNED SUNGEM', 134: 'HOUSE FINCH', 135: 'HOUSE SPARROW',
       136: 'IMPERIAL SHAQ', 137: 'INCA TERN', 138: 'INDIAN BUSTARD', 139: 'INDIAN PITTA', 140: 'INDIGO BUNTING',
       141: 'JABIRU', 142: 'JAVA SPARROW', 143: 'KAKAPO', 144: 'KILLDEAR', 145: 'KING VULTURE', 146: 'KIWI',
       147: 'KOOKABURRA', 148: 'LARK BUNTING', 149: 'LEARS MACAW', 150: 'LILAC ROLLER', 151: 'LONG-EARED OWL',
       152: 'MAGPIE GOOSE', 153: 'MALABAR HORNBILL', 154: 'MALACHITE KINGFISHER', 155: 'MALEO', 156: 'MALLARD DUCK',
       157: 'MANDRIN DUCK', 158: 'MARABOU STORK', 159: 'MASKED BOOBY', 160: 'MASKED LAPWING', 161: 'MIKADO  PHEASANT', 162: 'MOURNING DOVE', 163: 'MYNA', 164: 'NICOBAR PIGEON', 165: 'NOISY FRIARBIRD',
       166: 'NORTHERN BALD IBIS', 167: 'NORTHERN CARDINAL', 168: 'NORTHERN FLICKER', 169: 'NORTHERN GANNET',
       170: 'NORTHERN GOSHAWK', 171: 'NORTHERN JACANA', 172: 'NORTHERN MOCKINGBIRD', 173: 'NORTHERN PARULA',
       174: 'NORTHERN RED BISHOP', 175: 'NORTHERN SHOVELER', 176: 'OCELLATED TURKEY', 177: 'OKINAWA RAIL',
       178: 'OSPREY', 179: 'OSTRICH', 180: 'OYSTER CATCHER', 181: 'PAINTED BUNTIG', 182: 'PALILA',
       183: 'PARADISE TANAGER', 184: 'PARUS MAJOR', 185: 'PEACOCK', 186: 'PELICAN', 187: 'PEREGRINE FALCON',
       188: 'PHILIPPINE EAGLE', 189: 'PINK ROBIN', 190: 'PUFFIN', 191: 'PURPLE FINCH', 192: 'PURPLE GALLINULE',
       193: 'PURPLE MARTIN', 194: 'PURPLE SWAMPHEN', 195: 'QUETZAL', 196: 'RAINBOW LORIKEET', 197: 'RAZORBILL',
       198: 'RED BEARDED BEE EATER', 199: 'RED BELLIED PITTA', 200: 'RED BROWED FINCH', 201: 'RED FACED CORMORANT',
       202: 'RED FACED WARBLER', 203: 'RED HEADED DUCK', 204: 'RED HEADED WOODPECKER', 205: 'RED HONEY CREEPER',
       206: 'RED TAILED THRUSH', 207: 'RED WINGED BLACKBIRD', 208: 'RED WISKERED BULBUL', 209: 'REGENT BOWERBIRD',
       210: 'RING-NECKED PHEASANT', 211: 'ROADRUNNER', 212: 'ROBIN', 213: 'ROCK DOVE', 214: 'ROSY FACED LOVEBIRD',
       215: 'ROUGH LEG BUZZARD', 216: 'ROYAL FLYCATCHER', 217: 'RUBY THROATED HUMMINGBIRD', 218: 'RUFOUS KINGFISHER',
       219: 'RUFUOS MOTMOT', 220: 'SAMATRAN THRUSH', 221: 'SAND MARTIN', 222: 'SCARLET IBIS', 223: 'SCARLET MACAW',
       224: 'SHOEBILL', 225: 'SHORT BILLED DOWITCHER', 226: 'SMITHS LONGSPUR', 227: 'SNOWY EGRET', 228: 'SNOWY OWL',
       229: 'SORA', 230: 'SPANGLED COTINGA', 231: 'SPLENDID WREN', 232: 'SPOON BILED SANDPIPER', 233: 'SPOONBILL',
       234: 'SRI LANKA BLUE MAGPIE', 235: 'STEAMER DUCK', 236: 'STORK BILLED KINGFISHER', 237: 'STRAWBERRY FINCH',
       238: 'STRIPPED SWALLOW', 239: 'SUPERB STARLING', 240: 'SWINHOES PHEASANT', 241: 'TAIWAN MAGPIE', 242: 'TAKAHE',
       243: 'TASMANIAN HEN', 244: 'TEAL DUCK', 245: 'TIT MOUSE', 246: 'TOUCHAN', 247: 'TOWNSENDS WARBLER',
       248: 'TREE SWALLOW', 249: 'TRUMPTER SWAN', 250: 'TURKEY VULTURE', 251: 'TURQUOISE MOTMOT', 252: 'UMBRELLA BIRD',
       253: 'VARIED THRUSH', 254: 'VENEZUELIAN TROUPIAL', 255: 'VERMILION FLYCATHER', 256: 'VICTORIA CROWNED PIGEON',
       257: 'VIOLET GREEN SWALLOW', 258: 'VULTURINE GUINEAFOWL', 259: 'WATTLED CURASSOW', 260: 'WHIMBREL',
       261: 'WHITE CHEEKED TURACO', 262: 'WHITE NECKED RAVEN', 263: 'WHITE TAILED TROPIC', 264: 'WILD TURKEY',
       265: 'WILSONS BIRD OF PARADISE', 266: 'WOOD DUCK', 267: 'YELLOW BELLIED FLOWERPECKER', 268: 'YELLOW CACIQUE',
       269: 'YELLOW HEADED BLACKBIRD'}

map_dict1 = { 0: 'dog',
            1: 'horse',
            2: 'elephant',
            3: 'butterfly',
            4: 'chicken',
            5: 'cat',
            6: 'cow',
            7: 'sheep',
            8: 'spider',
            9: 'squirrel',
}

def welcome():
    st.title('WelCome to Birds and Animal '
                      +'Classification')
    st.image('animals_birds.png', use_column_width=True)
    st.subheader('A unique Web app that shows different Birds 269- Species Classification ,'
                +' Domestic Animals Classification And Cats OR Dogs Identification'
                 + ' You can choose the options from the left. It has a sidebar with '
                 +'Bird Species Classification, Domestio Animals Classification ,Some Birds and Animals Scientific Name')

    st.subheader('Made by Chumui')

def upload_file_work():
    st.header("You Chosen :- ")
    st.subheader("BIRDS 269 - SPECIES IMAGE CLASSIFICATION")
    st.image('logo_bird.jpg')
    uploaded_file = st.file_uploader("Choose a image file", type=[".jpg", ".jpeg", ".png"])

    model = tf.keras.models.load_model("model/BC.h5")

    if uploaded_file:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image, (224, 224))

        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="RGB")

        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = resized[np.newaxis, ...]

        Genrate_pred = st.sidebar.button("Generate Prediction")

        if Genrate_pred:
            prediction = model.predict(img_reshape).argmax()
            st.title("Predicted Label for the image is {}".format(map_dict[prediction]))

def domestic_animals():
    st.header("You Chosen :- ")
    st.subheader("Domestic Animals Classification")
    st.image('domestic_img1.png')

    uploaded_file = st.file_uploader("Choose a image file", type=[".jpg", ".jpeg", ".png"])

    model = tf.keras.models.load_model("mdl_wts.hdf5")

    if uploaded_file:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image, (224, 224))

        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="RGB")

        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = resized[np.newaxis, ...]

        Genrate_pred = st.sidebar.button("Generate Prediction")

        if Genrate_pred:
            prediction = model.predict(img_reshape).argmax()
            st.title("Predicted Label for the image is {}".format(map_dict1[prediction]))

def scie_birds():
    type_name = st.sidebar.selectbox("Scientific Name type", ["Indian Parrot","Indian eagle-owl","Indian spotted eagle","Indian Vulture",
    "Indian peafowl/peacock","Common Kingfisher","Common blackbird", "Indian Crows","Swans","Toucans", "WoodPecker","Penguins"
                        ])
    st.header("Birds/Scientific name : AVES")
    st.subheader("You Chosen :- " + type_name)
    col1 , col2 = st.columns(2)

    if type_name == "Indian Parrot":
        st.title("Scientify name of "+ type_name+ " is 	Psittacula krameri ")
        col1.image("image/in_parrot.jpg")
        col2.image("image/in_parrot2.jpg")
        st.title("About")
        st.subheader("The rose-ringed parakeet (Psittacula krameri), also known as the ring-necked parakeet (more commonly known as the Indian ringneck parrot), is a medium-sized parrot in the genus Psittacula, of the family Psittacidae. It has disjunct native ranges in Africa and the Indian Subcontinent, and is now introduced into many other parts of the world where feral populations have established themselves and are bred for the exotic pet trade.")
    elif type_name == "Indian eagle-owl":
        st.title("Scientify name of " + type_name + " is Bubo bengalensis ")
        col1.image("image/in_owl.jpg")
        col2.image("image/in_owl2.jpg")
        st.title("About")
        st.subheader("The Indian eagle-owl (Bubo bengalensis), also called the rock eagle-owl or Bengal eagle-owl, is a large horned owl species native to hilly and rocky scrub forests in the Indian Subcontinent. It is splashed with brown and grey, and has a white throat patch with black small stripes. It was earlier treated as a subspecies of the Eurasian eagle-owl. It is usually seen in pairs. It has a deep resonant booming call that may be heard at dawn and dusk.")

    elif type_name == "Indian spotted eagle":
        st.title("Scientify name of " + type_name + " Clanga hastata ")
        col1.image("image/eagle.jpg")
        col2.image("image/eagle2.jpg")
        st.title("About")
        st.subheader("The Indian spotted eagle (Clanga hastata) is a large bird of prey native to South Asia. Like all typical eagles, it belongs to the family Accipitridae. The typical eagles are often united with the buteos, sea eagles and other more heavyset Accipitridae, but more recently it appears as if they are less distinct from the more slender accipitrine hawks.")

    elif type_name == "Indian Vulture":
        st.title("Scientify name of " + type_name + " is Gyps indicus")
        col1.image("image/vulture.jpg")
        col2.image("image/vulture2.jpg")
        st.title("About")
        st.subheader("The Indian vulture (Gyps indicus) is an Old World vulture native to India, Pakistan and Nepal. It has been listed as Critically Endangered on the IUCN Red List since 2002, as the population severely declined. Indian vultures died of kidney failure caused by diclofenac poisoning. It breeds mainly on hilly crags in central and peninsular India.The slender-billed vulture Gyps tenuirostris in the northern part of its range is considered a separate species.")

    elif type_name == "Indian peafowl/peacock":
        st.title("Scientify name of " + type_name + " is Pavo cristatus ")
        col1.image("image/peekcock.jpg")
        col2.image("image/peekcock2.png")
        st.title("About")
        st.subheader("The Indian peafowl (Pavo cristatus), also known as the common peafowl, and blue peafowl, is a peafowl species native to the Indian subcontinent. It has been introduced to many other countries. Male peafowl are referred to as peacocks, and female peafowl are referred to as peahens")

    elif type_name == "Common Kingfisher":
        st.title("Scientify name of " + type_name + " is Alcedo atthis ")
        col1.image("image/kingfisher.jpg")
        col2.image("image/kingfisher2.jpg")
        st.title("About")
        st.subheader("The common kingfisher (Alcedo atthis), also known as the Eurasian kingfisher and river kingfisher, is a small kingfisher with seven subspecies recognized within its wide distribution across Eurasia and North Africa. It is resident in much of its range, but migrates from areas where rivers freeze in winter.")
    elif type_name == "Common blackbird":
        st.title("Scientify name of " + type_name + " is Turdus merula ")
        col1.image("image/Common_Blackbird.jpg")
        col2.image("image/Common_Blackbird2.jpg")
        st.title("About")
        st.subheader("The common blackbird (Turdus merula) is a species of true thrush. It is also called the Eurasian blackbird (especially in North America, to distinguish it from the unrelated New World blackbirds),or simply the blackbird where this does not lead to confusion with a similar-looking local species. It breeds in Europe, Asiatic Russia, and North Africa, and has been introduced to Australia and New Zealand.It has a number of subspecies across its large range; a few of the Asian subspecies are sometimes considered to be full species. Depending on latitude, the common blackbird may be resident, partially migratory, or fully migratory.")
    elif type_name == "Indian Crows":
        st.title("Scientify name of " + type_name + " is Corvus splendens ")
        col1.image("image/crow.jpg")
        col2.image("image/crow2.jpg")
        st.title("About")
        st.subheader("The house crow (Corvus splendens), also known as the Indian, greynecked, Ceylon or Colombo crow,[2] is a common bird of the crow family that is of Asian origin but now found in many parts of the world, where they arrived assisted by shipping. It is between the jackdaw and the carrion crow in size (40 cm (16 in) in length) but is slimmer than either. The forehead, crown, throat and upper breast are a richly glossed black, whilst the neck and breast are a lighter grey-brown in colour. The wings, tail and legs are black. There are regional variations in the thickness of the bill and the depth of colour in areas of the plumage.")

    elif type_name == "Swans":
        st.title("Scientify name of " + type_name + " is Cygnus ")
        col1.image("image/Swan.jpg")
        col2.image("image/Swan2.jpg")
        st.title("About")
        st.subheader("Swans are birds of the family Anatidae within the genus Cygnus. The swans' closest relatives include the geese and ducks. Swans are grouped with the closely related geese in the subfamily Anserinae where they form the tribe Cygnini. Sometimes, they are considered a distinct subfamily, Cygninae. There are six living and many extinct species of swan; in addition, there is a species known as the coscoroba swan which is no longer considered one of the true swans.")

    elif type_name == "Toucans":
        st.title("Scientify name of " + type_name + " is Ramphastidae ")
        col1.image("image/toucan.jpg")
        col2.image("image/toucan2.png")
        st.title("About")
        st.subheader("Toucans are members of the Neotropical near passerine bird family Ramphastidae. The Ramphastidae are most closely related to the American barbets. They are brightly marked and have large, often colorful bills. The family includes five genera and over forty different species.Toucans are arboreal and typically lay 2–21 white eggs in their nests.")

    elif type_name == "WoodPecker":
        st.title("Scientify name of " + type_name + " is Picidae ")
        col1.image("image/woodpecker.jpg")
        col2.image("image/woodpecker2.jpg")
        st.title("About")
        st.subheader("Woodpeckers are part of the family Picidae, which also includes the piculets, wrynecks, and sapsuckers. Members of this family are found worldwide, except for Australia, New Guinea, New Zealand, Madagascar, and the extreme polar regions. Most species live in forests or woodland habitats, although a few species are known that live in treeless areas, such as rocky hillsides and deserts, and the Gila woodpecker specialises in exploiting cacti.")

    elif type_name == "Penguins":
        st.title("Scientify name of " + type_name + " is Spheniscidae ")
        col1.image("image/penguin.jpg")
        col2.image("image/penguin3.jpg")
        st.title("About")
        st.subheader("Penguins (order Sphenisciformes, family Spheniscidae) are a group of aquatic flightless birds. They live almost exclusively in the Southern Hemisphere: only one species, the Galápagos penguin, is found at or north of the Equator. Highly adapted for life in the water, penguins have countershaded dark and white plumage and flippers for swimming. Most penguins feed on krill, fish, squid and other forms of sea life which they catch with their bills and swallow it whole while swimming. A penguin has a spiny tongue and powerful jaws to grip slippery prey.")
    else:
        st.title("Choose right option")

def scie_animal():
    type_name = st.sidebar.selectbox("Scientific Name of ", [ "Cats",
                                                              "Dogs" , "Cows" ,
                                                              "Tigers", "Lions",
                                                              "Horse","Old World Monkey" ,
                                                              "Elephants" , "Giraffe" , "Bears"
    ])

    st.header("Animal/Scientific name : ANIMALIA")
    st.subheader("You Chosen :- " + type_name)
    col1, col2 = st.columns(2)

    if type_name == "Cats":
        st.title("Scientify name of " + type_name + " is Felis catus ")
        col1.image("image_animal/cat.jpg")
        col2.image("image_animal/cat2.jpg")
        st.title("Audio")
        audio_fie = open("audio/cat_audio.wav", 'rb')
        audio_bytes = audio_fie.read()

        st.audio(audio_bytes, format='audio/wav')

        st.title("About")
        st.subheader("The cat (Felis catus) is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae and is often referred to as the domestic cat to distinguish it from the wild members of the family.")
        st.subheader("Lifespan: 12 – 18 years (Domesticated) , Gestation period: 58 – 67 days , Scientific name: Felis catus , Daily sleep: 12 – 16 hours , Speed: 48 km/h (Maximum), Mass: 4 – 5 kg (Domesticated)")

    elif type_name == "Dogs":
        st.title("Scientify name of " + type_name + " is Canis lupus familiaris ")
        col1.image("image_animal/dog1.jpg")
        col2.image("image_animal/dog3.jpg")
        st.title("Audio")
        audio_fie = open("audio/dog_audio.wav", 'rb')
        audio_bytes = audio_fie.read()

        st.audio(audio_bytes, format='audio/wav')
        st.title("About")
        st.subheader("The dog or domestic dog (Canis familiaris or Canis lupus familiaris) is a domesticated descendant of the wolf, and is characterized by an upturning tail. The dog is derived from an ancient, extinct wolf, and the modern wolf is the dog's nearest living relative.[8] The dog was the first species to be domesticated, by hunter–gatherers over 15,000 years ago,before the development of agriculture.")
        st.subheader("Lifespan: 10 – 13 years, Scientific name: Canis lupus familiaris, Gestation period: 58 – 68 days, Height: 15 – 110 cm (At Shoulder), Daily sleep: 12 – 14 hours (Adult), Speed: German Shepherd - 48 km/h, Greyhound: 72 km/h")

    elif type_name == "Cows":
        st.title("Scientify name of " + type_name + " is Bos taurus ")
        col1.image("image_animal/cow.jpg")
        col2.image("image_animal/cow2.jpg")
        st.title("Audio")
        audio_fie = open("audio/cow_audio.wav", 'rb')
        audio_bytes = audio_fie.read()

        st.audio(audio_bytes, format='audio/wav')
        st.title("About")
        st.subheader("Cattle (Bos taurus) are large domesticated bovines. They are most widespread species of the genus Bos. Adult females are referred to as cows and adult males are referred to as bulls.")
        st.subheader("Scientific name: Bos taurus Gestation, period: 283 days ,Daily sleep: 4 hours (Female, Adult, Cow),Speed: 40 km/h (Maximum), Higher classification: Bos , Mass: 1,100 kg (Male, Adult, Bull), 720 kg (Female, Adult, Cow)")

    elif type_name == "Tigers":
        st.title("Scientify name of " + type_name + " is Panthera tigris ")
        col1.image("image_animal/tiger.jpg")
        col2.image("image_animal/tiger2.jpg")
        st.title("Audio")
        audio_fie = open("audio/tiger_audio.mp3", 'rb')
        audio_bytes = audio_fie.read()

        st.audio(audio_bytes, format='audio/mp3')
        st.title("About")
        st.subheader("The tiger is the largest living cat species and a member of the genus Panthera. It is most recognisable for its dark vertical stripes on orange fur with a white underside. An apex predator, it primarily preys on ungulates such as deer and wild boar.")
        st.subheader("Lifespan: 8 – 10 years (In the wild), Scientific name: Panthera tigris, Speed: 49 – 65 km/h (In Short Bursts) , Conservation status: Endangered (Population decreasing) Encyclopedia of Life, Height: 80 – 110 cm (At Shoulder), Mass: 90 – 310 kg (Male, Adult), 65 – 170 kg (Female, Adult)")

    elif type_name == "Lions":
        st.title("Scientify name of " + type_name + " is Panthera leo ")
        col1.image("image_animal/lion2.jpg")
        col2.image("image_animal/lion.png")
        st.title("Audio")
        audio_fie = open("audio/lion_audio.wav", 'rb')
        audio_bytes = audio_fie.read()

        st.audio(audio_bytes, format='audio/wav')
        st.title("About")
        st.subheader("The lion is a large cat of the genus Panthera native to Africa and India. It has a muscular, broad-chested body, short, rounded head, round ears, and a hairy tuft at the end of its tail. It is sexually dimorphic; adult male lions are larger than females and have a prominent mane")
        st.subheader("Scientific name: Panthera leo,Speed: 80 km/h (Maximum, In Short Bursts),Trophic level: Carnivorous Encyclopedia of Life,Lifespan: 15 – 16 years (Female, Adult, In the wild), 8 – 10 years (Male, Adult, In the wild),Mass: 190 kg (Male, Adult), 130 kg (Female, Adult),Height: 1.2 m (Male, Adult, At Shoulder), 90 – 110 cm (Female, Adult, At Shoulder)")

    elif type_name == "Horse":
        st.title("Scientify name of " + type_name + " is Equus caballus ")
        col1.image("image_animal/horse2.jpg")
        col2.image("image_animal/horse4.jpg")
        st.title("Audio")
        audio_fie = open("audio/horse_audio.wav", 'rb')
        audio_bytes = audio_fie.read()

        st.audio(audio_bytes, format='audio/wav')
        st.title("About")
        st.subheader("The horse is a domesticated, odd-toed, hoofed mammal. It belongs to the taxonomic family Equidae and is one of two extant subspecies of Equus ferus. The horse has evolved over the past 45 to 55 million years from a small multi-toed creature, Eohippus, into the large, single-toed animal of today.")
        st.subheader("Lifespan: 25 – 30 years, Speed: 88 km/h (Maximum, Sprint) ,Scientific name: Equus caballus ,Mass: 300 kg (Adult) Encyclopedia of Life ,Gestation period: 11 – 12 months ,Height: Arabian horse: 1.4 – 1.6 m, Hanoverian horse: 1.6 – 1.7 m, Akhal-Teke: 1.5 – 1.6 m")

    elif type_name == "Old World Monkey":
        st.title("Scientify name of " + type_name + " is Cercopithecidae ")
        col1.image("image_animal/monkey.jpg")
        col2.image("image_animal/monkey1.jpg")
        st.title("Audio")
        audio_fie = open("audio/monkey_audio.wav", 'rb')
        audio_bytes = audio_fie.read()

        st.audio(audio_bytes, format='audio/wav')
        st.title("About")
        st.subheader("Old World monkey is the common English name for a family of primates known taxonomically as the Cercopithecidae. Twenty-four genera and 138 species are recognized, making it the largest primate family. Old World monkey genera include baboons, red colobus and macaques.")
        st.subheader("Scientific name: Cercopithecidae,Higher classification: Cercopithecoidea,Rank: Family,Lifespan: Japanese macaque: 27 years, Mandrill: 20 years, Lion-tailed macaque: 20 years, Guinea baboon: 35 – 45 years,Gestation period: Rhesus macaque: 166 days,Height: Japanese macaque: 57 cm, Mandrill: 55 – 65 cm, Olive baboon: 70 cm")

    elif type_name == "Elephants":
        st.title("Scientify name of " + type_name + " is Elephantidae ")
        col1.image("image_animal/elephant1.jpg")
        col2.image("image_animal/elephant2.jpg")
        st.title("Audio")
        audio_fie = open("audio/elephant_audio.mp3", 'rb')
        audio_bytes = audio_fie.read()

        st.audio(audio_bytes, format='audio/mp3')
        st.title("About")
        st.subheader("Elephantidae is a family of large, herbivorous proboscidean mammals collectively called elephants and mammoths. These are terrestrial large mammals with a snout modified into a trunk and teeth modified into tusks. Most genera and species in the family are extinct. Only two genera, Loxodonta and Elephas, are living.")
        st.subheader("Scientific name: Elephantidae,Gestation period: African bush elephant: 22 months,Height: African bush elephant: 3.2 m,Speed: African bush elephant: 40 km/h,Rank: Family,Lifespan: African bush elephant: 60 – 70 years, African elephants: 33 years")

    elif type_name == "Giraffe":
        st.title("Scientify name of " + type_name + " is Giraffa ")
        col1.image("image_animal/giraffe.jpg")
        col2.image("image_animal/giraffe2.jpg")
        st.title("Audio")
        audio_fie = open("audio/giraffe_audio.mp3", 'rb')
        audio_bytes = audio_fie.read()

        st.audio(audio_bytes, format='audio/mp3')
        st.title("About")
        st.subheader("The giraffe is a tall African hoofed mammal belonging to the genus Giraffa. It is the tallest living terrestrial animal and the largest ruminant on Earth. Traditionally, giraffes were thought to be one species, Giraffa camelopardalis, with nine subspecies.")
        st.subheader("Eats: Acacia,Scientific name: Giraffa,Gestation period: Northern giraffe: 15 months,Daily sleep: 4.6 hours (In captivity),Speed: 60 km/h (Maximum, Sprint),Height: 5 – 5.9 m (Male, Adult), 4.3 – 5.2 m (Female, Adult)")

    elif type_name == "Bears":
        st.title("Scientify name of " + type_name + " is Ursidae ")
        col1.image("image_animal/bear.jpg")
        col2.image("image_animal/bear2.jpg")
        st.title("Audio")
        audio_fie = open("audio/bear_audio.mp3", 'rb')
        audio_bytes = audio_fie.read()

        st.audio(audio_bytes, format='audio/mp3')
        st.title("About")
        st.subheader("Bears are carnivoran mammals of the family Ursidae. They are classified as caniforms, or doglike carnivorans. Although only eight species of bears are extant, they are widespread, appearing in a wide variety of habitats throughout the Northern Hemisphere and partially in the Southern Hemisphere.")
        st.subheader("Scientific name: Ursidae,Lifespan: Giant panda: 20 years, Brown bear: 20 – 30 years,Speed: Polar bear: 40 km/h, Brown bear: 56 km/h,Height: Polar bear: 1.8 – 2.4 m, Giant panda: 60 – 90 cm, Brown bear: 70 – 150 cm,Gestation period: Polar bear: 195 – 265 days, MORE Encyclopedia of Life,Tail length: Polar bear: 7 – 13 cm, Giant panda: 10 – 15 cm, Brown bear: 6 – 22 cm")

def main():
    st.sidebar.image("main_logo.png")
    st.sidebar.title("Birds & Animals Classifcation")
    function_selected = st.sidebar.selectbox("Choose function",
                                             [ "WelCome",
                                              "Bird Species Classification",
                                              "Domestic Animals Classification",
                                               "Birds Scientific Name",
                                               "Animals Scientific Name"])
    if function_selected == "WelCome":
        welcome()
    elif function_selected == "Bird Species Classification":
        upload_file_work()
    elif function_selected == "Domestic Animals Classification":
        domestic_animals()
    elif function_selected == "Birds Scientific Name":
        scie_birds()
    elif function_selected == "Animals Scientific Name":
        scie_animal()
    else:
        st.title("Choose right option")


if __name__ == "__main__":
    main()




