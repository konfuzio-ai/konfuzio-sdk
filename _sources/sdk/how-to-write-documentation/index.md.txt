## How to write documentation?

### Guidelines for Sphinx Documentation

If you're working on the SDK and need to update the documentation, here are some guidelines on how to add new pages and manage assets such as images in our Sphinx documentation.

### Adding New Content and Pages

When you want to add new content or create a new page:

1.  Create a separate folder for your new content in the appropriate directory within the `docs` folder. The name of this new folder will form part of the URL for the new page.

2.  Within this new folder, create a new `index.rst` (reStructuredText) or `index.md` (Markdown) file. This file will be the main page for the new content.

3.  Write your content in the `index` file using the appropriate syntax. If you're new to reStructuredText, check out this [quick reference](https://docutils.sourceforge.io/docs/user/rst/quickref.html).

4.  To make your new page discoverable, add it to the appropriate `toctree` directive in an existing `.rst` file that's one level up in the directory structure. For example, if your new page is a subsection of an existing page, you would add it to that page's `toctree`.

### Organizing Images and Other Assets

When adding images or other assets to the documentation:

1.  Place all images or assets related to your new content within the same folder you created in step 1 above. This helps keep our documentation well organized and makes it easier for other contributors to find and update assets related to specific content.

2.  When referencing an image in your documentation, use a relative path from the current file to the image. For example: `.. image:: my-image.png`

3.  If you're adding the image through the GitHub web editor and it automatically creates a different assets folder, please manually move the image to the correct folder as described above.

Remember, clear and organized documentation makes it easier for users to understand and effectively use our SDK. Thank you for your contribution to improving our documentation!

After adding this section, the next steps would be:

-   Add the guidelines to the relevant section of the Documentation.
-   Open a PR for your changes.
-   Ensure that the changes are correctly reflected in the Sphinx documentation.
-   Merge the PR after review.
