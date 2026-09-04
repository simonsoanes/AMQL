# AMQL - C# implementation of VIndex3 (Larql)

This is a port of the VIndex3 implementation, along with support for generating it from a model (Qwen 3.5 initially) and then allowing model independent inference, token relationship route following and exploration of the model internals in order to do some research into direct model manipulation and patching, with live LORA adapters in custom inferencing.

Credit for the design of the VIndex3 goes to Chris Hay.

## What is this for?

When building an continuous cognition platform, I had the issue that all current LLM models have flaws and no way to to self-improve.  This project is intended to eventually be a tool for AI self-improvement.

This project allows you to turn a model into a graph database, then query the relationship between tokens (or their text representations).  Once it has identified the edges in the graph, it is then possible to generate a specific LORA adapter to adjust that behaviour in the model (or re-write the base model), effectively editing the input and output knowledge.

At the current level, it's possible to remove the concept of something being associated in a particular way, or add a new association between two items in a familiar way (PlaceA is capital of PlaceB) - this is helpful to correct flaws in the embedding layer.  It's also possible to interact with a relationship between two things - where there is a relationship but it's of the incorrect type.  

Looking at the next layer of abstraction it's also possible to adjust relationships that humanity hasn't described linguistically yet but infers a connection between normally.  This has benefits when combined with positive and negative re-inforcement derived from internal traces taken during the inference stage in a model.  The intent here being to correct things like halucinations (where the relationship is 'user satisfaction' added in post-training) to incorrect selections of tools when operating agentically.

The hope is that it's eventually possible calculate the representation an approach to a problem space - and potentially to create new ones or transpose them - for example, allowing the application of first order logic to an understanding where currently it's not completely trained into the model because of the lack of input source or smaller model density.

Combined with this being an automatic self-learning process, it should increase the models understanding and intelligence beyond currently trainable human textual representations when it's apparent that a known construct would be better applied then - something that can't be achieved through fine-tuning without generating the more intelligent outcome scenarios, which are therefore inherently gated at human-intellect.

The generation of novel solution vectors to a problem space is a separate distinct challenge requiring a new approach, but this does allow applying alternate solution vectors when they have been identified.