---
title: "Blog"
layout: archive
permalink: /posts/
---

{% if paginator %}
  {% assign posts = paginator.posts %}
{% else %}
  {% assign posts = site.posts %}
{% endif %}

<script type="text/javascript">
    function filterUsingCategory(selectedCategory) {
      {% for post in posts %}
        var cats = {{ post.tags | jsonify }}

        var postDiv = document.getElementById("post-{{post.title | slugify}}");
        postDiv.style.display = (selectedCategory == 'All' || cats.includes(selectedCategory))
          ? 'unset'
          : 'none';
      {% endfor %}
    }
</script>

  <div class="btn-group">
    <button id="All" class="button-71" role="button" onclick="filterUsingCategory('All')">All ({{ posts.size }})</button>
    {% assign tags = site.tags | sort %}
    {% for category in tags %}
      {% assign cat = category | first %}
      <button id="{{ cat }}" class="button-71" role="button" onclick="filterUsingCategory(this.id)">{{ cat }} ({{ site.tags[cat].size }})</button>
    {% endfor %}
    <hr />
  </div>

  <div class="posts-wrapper">
    {% for post in posts %}
      <div class="post" id="post-{{post.title | slugify}}">
        <p class="itemInteriorSection">
          {%- unless post.hidden -%}
            {% include archive-single.html %}
            {% if post.image %}
              <a href="{{ post.link }}"><img src="{{ post.image }}"></a>
            {% endif %}
          {%- endunless -%}
        </p>
      </div>
    {% endfor %}
  </div>
