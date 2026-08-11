# Open Weights Are Not the Same as Open Source

One of the most common licensing mistakes in applied machine learning is treating
a downloadable model as an open-source model. These are different things, and the
difference has commercial consequences.

## What open source means

An open-source licence, in the sense recognised by the Open Source Initiative,
permits use for any purpose, by any person or group, in any field of endeavour,
without additional restriction. The Apache 2.0 and MIT licences meet this bar.
Software under them can be used commercially, modified, and redistributed without
asking permission or meeting a usage threshold.

## What open weights means

Open weights means only that the trained parameters of a model are available for
download. The licence attached to those weights may impose restrictions that an
open-source licence would not permit.

Meta's Llama models are the clearest example. Llama 3.x is distributed under the
Llama Community Licence, not under Apache 2.0 or MIT. That licence carries an
acceptable use policy restricting certain applications, and it contains a clause
requiring organisations with more than 700 million monthly active users at the
time of release to request a separate licence from Meta. It also imposes naming
and attribution requirements on derivative models. These are reasonable terms,
but they are not open source, and describing Llama as open source is inaccurate.

Llama Guard, the safety classifier model, is distributed under the same community
licence and inherits the same restrictions.

## Models that are genuinely open source

Several strong models are released under true open-source licences. Mistral has
published models under Apache 2.0, including Mistral 7B. Qwen has released
several model sizes under Apache 2.0. DeepSeek has published models under the MIT
licence. Nomic Embed is released under Apache 2.0, which is part of why it is a
comfortable default for an embedding layer.

## Why this matters for a zero-cost stack

A stack can be free to run and still be commercially restricted. Running a model
locally removes the vendor bill; it does not remove the licence. Before any
commercial deployment, the licence of each specific model version should be read
directly, because model families sometimes change licence between releases and
the restrictions differ by model rather than by publisher.

The practical rule is to separate two questions. First: does this cost money to
run? Second: am I permitted to use it for this purpose? A locally hosted Llama
model answers yes to the first and requires a careful reading for the second.
