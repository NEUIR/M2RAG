
def get_prompt(task,topk):
    if topk==0:
        if task=='image_cap':
            prompt='''You are an intelligent assistant capable of generating accurate and detailed captions for images.
I will provide you with an image.
Response the caption for the image directly, do not include any explanations or unrelated details.

Caption:'''
        elif task=='mmqa':
            prompt='''You are an intelligent assistant capable of answering complex questions.
Your task is to carefully analyze the queation and provide a detailed, well-structured, and contextually accurate answer in the form of a complete sentence. 
Write a detailed and accurate answer directly, do not include unrelated details in your response.

Question: {question}

Answer:'''
        elif task=='fact_verify':
            prompt='''You are an intelligent assistant capable of verifying the factual accuracy by classifying data samples into one of three categories: Support, Insufficient, and Refute.
I will provide you with a claim (to be verified) consisting of text and an image.
Your task is to verify the claim based on your own knowledge into one of the three categories.
Response to the category of the claim directly, do not say any other words or explain.

Claim_Text: {claim_text}

Category:'''

    else:
        if task=='image_cap':
            prompt='''You are an intelligent assistant capable of generating accurate and detailed captions for images.
You will be given one image and several retrieved images and their captions. The first image is the input image, others are retrieved examples to help you.
Response the caption for the image directly, do not include any explanations or unrelated details.

Image: The first image.
Retrieved_Image_Caption: {retrirval_image_caption}

Caption:'''
        elif task=='mmqa':
            prompt='''You are an intelligent assistant capable of answering complex questions using both visual and textual data.
I will provide you with a question and several retrieved images or texts to assist you.
Your task is to carefully analyze the queation and retrieved information to provide a detailed, well-structured, and contextually accurate answer in the form of a complete sentence. 
Write a detailed and accurate answer directly, do not include additional context, or unrelated details in your response.

Question: {question}
Retrieved_Image_Caption: {retrirval_image_caption}
Retrieved_Text: {retrirval_text}

Answer:'''
        elif task=='fact_verify':
            prompt='''You are an intelligent assistant capable of verifying the factual accuracy by classifying data samples into one of three categories: Support, Insufficient, and Refute.
I will provide you with a claim (to be verified) consisting of text and an image, and retrieved images or texts. The first image is the claim_image, and others are retrieved examples to help you.
Your task is to classify the relationship between the claim and the evidence from the retrieved documents and your own knowedge into one of the three categories.
Response the category for the claim and document directly, do not say any other word or explain.

Claim_Image: The first image.
Claim_Text: {claim_text}
Retrieved_Document_Text: {retrirval_text}

Category:'''
        
    return prompt