/**
 *Submitted for verification at Etherscan.io on 2017-10-21
*/

pragma solidity ^0.4.15;

contract AffiliateNetwork {
    uint public idx = 0;
    mapping (uint => address) public affiliateAddresses;
    mapping (address => uint) public affiliateCodes;

    function () payable {
        if (msg.value > 0) {
            msg.sender.transfer(msg.value);
        }

        addAffiliate();
    }

    function addAffiliate() {
        if (affiliateCodes[msg.sender] != 0) { return; }

        idx += 1;   // first assigned code will be 1
        affiliateAddresses[idx] = msg.sender;
        affiliateCodes[msg.sender] = idx;
    }
}