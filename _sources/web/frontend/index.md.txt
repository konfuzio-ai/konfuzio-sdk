# Frontend

## HTML iframe as Public Document

You can mark your documents as *public*. Marking documents as public will generate a **read-only**, **publicly accessible view**, that does not require authentication. This allows you to share a link to the document and its extracted data, or embed it in another website.

### Mark a document as public

1. Select your document from the document list on the Konfuzio dashboard.
2. Click on the "Details" link near the document name in the top bar.
3. Check the "Is public" checkbox.
4. Click "Save and continue editing".
5. After saving, new options will appear at the bottom of this page.

### Share a document with a link

From the details page, you can copy a public URL to your document, which you can share with other people. Apart from the URL, it does not contain any Konfuzio branding.

This lightweight version contains an image version of the PDF pages, and its currently extracted metadata (annotation sets, label sets, labels). Any modification you make to the document in the SmartView or TextView is reflected here.

Currently this public view is not allowed to be indexed by search engines.

If you need to programmatically generate public links, you can use the format `https://app.konfuzio.com/d/<id>/`. You can retrieve the `id` of a document from your Konfuzio dashboard or the API. Document `id`s which don't exist or are not public will return a 404.

### Embed a Konfuzio document on another website

From the details page, you can copy an example HTML snippet that allows you to embed a public document within an `iframe`. Visually, it looks the same as the above-mentioned public document view, and in fact its internal implementation is the same. However, to prevent abuse, you first need to configure your project's "domain whitelist" setting. This only needs to be done *once* per project for each domain you want to allow.

#### Add your domain(s) to the project's domain whitelist

1. On the Konfuzio dashboard, open the left-side menu and click "Projects".
2. Click on the project associated to the document(s) you want to make public.
3. In the "Domain whitelist" field, add the domains where you're going to embed your document(s), one per line and without "http" or "https".
   * For example, if you want to embed a document on `https://www.example.org`, you should add `www.example.org` to the list.
4. Click "Save".

```{admonition} Note
:class: important
This process **will NOT** make all your project's documents public by default. It simply estabilishes which domains are allowed to embed public documents for this project. You will still need to mark documents as public by yourself.
```

#### Customize the `iframe`

By default, we provide a bare-bones HTML snippet that looks similar to this:

```html
<iframe src="https://app.konfuzio.com/d/<id>/" width="100%" height="600" frameborder="0"></iframe>
```

This creates on your page an `iframe` with 100% width (full width of its container) and a height of 600 pixels, that doesn't have a border. You can customize the `iframe`'s size and other options by changing these and other attributes (see [iframe documentation](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe)).

Currently you cannot customize the style (fonts, color, etc.) of the `iframe`'s content. While we plan to offer more customization options in the future, advanced users might want to take a look at our [create your own dashboard guide](https://dev.konfuzio.com).

  
## Integration via Capture Vue
  
See [GitHub](https://github.com/konfuzio-ai/konfuzio-capture-vue#documentation)
