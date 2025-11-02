---
layout: about
title: about
nav: true
nav_order: 3
permalink: /about
subtitle: <a href='#'>about me</a>

profile:
  align: right
  image: about.jpg
  image_circular: false # crops the image to make it circular
  address: >

news: false  # includes a list of news items
social: true  # includes social icons at the bottom of the page
---

### Experientia Docet

Hi,
I'm an AI Solutions Architect focusing on Instinct GPU at AMD.
I'm passionate about AI, Accelerated Heterogeneous Computing, HPC, and GPU programming.

Previously, I was an LLM engineer at NAVER and a Solutions Architect at NVIDIA.
I also authored a CUDA programming book, "Learn CUDA Programming", with Bharatkumar Sharma, published by Packt Publishing.

<!-- CV -->
<details>
  <article>
    <div class="cv">
      {% for entry in site.data.cv %}
        <div class="card mt-3 p-3">
          <h3 class="card-title font-weight-medium">{{ entry.title }}</h3>
          <div>
          {% if entry.type == "list" %}
            {% include cv/list.liquid %}
          {% elsif entry.type == "map" %}
            {% include cv/map.liquid %}
          {% elsif entry.type == "nested_list" %}
            {% include cv/nested_list.liquid %}
          {% elsif entry.type == "time_table" %}
          {% include cv/time_table.liquid %}
          {% else %}
            {{ entry.contents }}
          {% endif %}
          </div>
        </div>
      {% endfor %}
      </div>
  </article>
</details>
