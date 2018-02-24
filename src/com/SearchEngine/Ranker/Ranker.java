package com.SearchEngine.Ranker;

import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.HashMap;

public class Ranker {

    /* The number of pages in the graph*/
    private Integer pagesCount;

    /* The Graph adjacency list */
    private HashMap<Integer, ArrayList<Integer>> inList = new HashMap<Integer, ArrayList<Integer>>();

    /* The out degrees of each page */
    private ArrayList<Integer> outDegrees;

    /* The ranks of the pages */
    private ArrayList<Double> pagesRank;

    /* The dumping factor */
    private final Double alpha = 0.85;

    /* The maximum number iterations */
    private final Integer maxIterations = 100;

    private void addArc(int from, int to) {
        if (inList.containsKey(to)) {
            inList.get(to).add(from);
        } else {
            ArrayList<Integer> newList = new ArrayList<Integer>();
            newList.add(from);
            inList.put(to, newList);
        }
        outDegrees.set(from, outDegrees.get(from) + 1);
    }

    private void initializeLists(int pagesCount) {
        this.pagesCount = pagesCount;

        outDegrees = new ArrayList<Integer>();
        pagesRank = new ArrayList<Double>();

        for (int i = 0; i < pagesCount; i++) {
            outDegrees.add(0);
            pagesRank.add(1.0/pagesCount);
        }
    }

    public void readFile(String filePath) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(filePath));
            String strLine;

            initializeLists(5000);

            // Read file line by line
            while ((strLine = br.readLine()) != null) {

                String[] strs = strLine.trim().split(" ");
                Integer u = Integer.parseInt(strs[0]);
                Integer v = Integer.parseInt(strs[1]);

                // Add arcs
                this.addArc(u, v);

                System.out.println(u.toString() + " " + v.toString());
            }
        } catch (Exception e) {
            //Catch exception if any
            System.err.println("Error: " + e.getMessage());
        }

        debugging();
    }

    public void rankPages() {
        // ToDo: @Samir55 Get the root page.

        // Check whether the node is dampling node or not.

        // Loop over all pages
        for (int page = 0; page < pagesCount; page++) {

            Double h_page = 0.0;
            if (inList.containsKey(page)) {
                for (Integer from : inList.get(page)) {
                    h_page += (1.0 * pagesRank.get(from) / outDegrees.get(from));
                }
                h_page *= alpha; // Multiply by dumping factor.
            }

//            Double ;

            pagesRank.set(page, (h_page));
        }

    }

    // Debugging @Samir55
    private void debugging() {
        System.out.println("The saved edges printing");

        for (Integer vertex : inList.keySet()) {
            System.out.println("For the vertex " + vertex.toString());
            System.out.println("OutDegrees " + outDegrees.get(vertex).toString());
            for (Integer to : inList.get(vertex)) {
                System.out.print(to.toString() + " ");
            }
            System.out.println();
        }
    }

}
